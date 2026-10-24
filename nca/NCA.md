# The NCA: long division as a shader

## In plain words

**The question.** A shader runs one small program at every pixel of a picture: millions of identical copies, each
looking only at its own neighbourhood, and the image comes out of all of them together. Could arithmetic be learned
that way? Take a grid of identical tiny cells; let every cell run the same small learned rule on its 3 × 3
neighbourhood; repeat, tick after tick, until the answer appears. The loops of the algorithm would come from the
repetition, not from anything written for them. This is the far end of the pattern that worked in this project, one
small digit cell reused at every place, and the opposite of a dense network, where every unit touches every other. We
asked whether it learns long division and then the square x² mod N, at the project's tiniest size, and how it fails.

**A tiny example.** Base 3, N = 5 and v = 14, written lowest digit first as `2 1` and `2 1 1`. The grid is the
worksheet of the long division, one row per step and one column per place:

```
            place 0 1 2 3
the prompt        2 1 1 0   v = 14, and N = 5 in a second channel
SHIFT             0 2 1 1   the register (places 1-3) is 14
SUB               0 0 0 1   14 - 5 = 9: no borrow
SEL               0 0 0 1   so keep 9
SUB               0 1 1 0   9 - 5 = 4: no borrow
SEL               0 1 1 0   so keep 4: the answer
```

Every cell does one digit's share of one row. A SUB cell subtracts N's digit, which has ridden down to it cell by
cell from the prompt, and the borrow handed over by the place below; it hands its own borrow to the place above. A SEL
cell passes the top place's borrow back along its row, and keeps or replaces its digit by it. Whether the scratch
cells start blank or random, the answer is right after 13 ticks.

**1. The format can do it exactly.** One rule, a small network of 56 hidden units at base 3 that every cell runs,
computes the division exactly at every size checked: bases 2 to 5, up to 4 digits of N and 4 quotient digits, every
cell of every question, whether the scratch cells start blank or random, and still after twice the ticks. 242 units do
x² mod N the same way, multiplying digit by digit before dividing. A bigger number needs no new weights, only a bigger
grid: the loops really do come from repetition.

**2. Taught the way this project teaches, it never learns, not even at the tiniest size.** 84 runs in
41 variants, with every lesson the format can express: the answers, x mod N = x, the one-link check, the
subtraction cell's own order laws; blurred or snapped digits; lessons staged, the sub-grid first; the cell reading
itself or not, reading downward or not; N's digits handed to every cell directly, or every long-range wire of the
bridge's machine. None certifies, and only 3 ever get all their training questions right. What they do
instead is memorize:
- with blurred digits the grid often fits all 99 training questions while its snapped version gets a third to a half,
  and one size up it is no better than "always answer 0": an analog code, as the bridge's loose networks had;
- with snapped digits it can fit training exactly with a lookup, and is at chance one size up;
- the subtraction cell's own laws make its table exact, but the grid around it never forms. Carrying N down the grid,
  remembering the old register, passing the borrow back along a row: these are copies through many cells, and nothing
  checks a copy where it happens.

**3. Told every move, it learns in a few hundred steps, and then growing the grid is the problem.** Labelling every
cell at every tick with the exact rule's own next state (the machine's trace, far more help than the task gives),
the training grid is learned in 250 to 850 steps and stays exact however long it runs. One size up it first failed,
for two reasons the diagnosis separates:
- **a second resting state.** The learned rule also read the row below, so rows fed back into each other and a bigger
  grid could settle into another self-consistent pattern. Hiding the row below, so that information flows only down
  and along rows, as on paper, fixes that;
- **situations never met.** A bigger grid puts cells next to neighbours they never had in training: with one more
  quotient digit, a SHIFT row under a SEL row; with one more digit of N, an inner place between two inner places. A
  third to two thirds of a bigger grid's cells see a neighbourhood training never showed, although almost none see
  anything new among the few inputs the exact rule reads. The learned rule reads everything.

With both fixed (information flowing one way, and a training grid of (3, 3, 2) in which every local situation occurs)
the learned rule **certifies**: in all 6 seeds checked on samples of the bigger sizes, and in 4 of 8 checked on every
question of them (8,505 and 78,003 questions). Every one of the 14 is exact at the width it was trained on, for every
number of quotient digits checked, and the best is exact out to (3, 5, 5), two more digits of N and three more quotient
digits than it ever saw (two of the six verified that far are). The others miss 0.03–1.1% of the questions one digit of
N wider, all on a few moduli whose digit patterns never occur at the training width, where a row settles into a second
resting state: the same trap one level down.

**4. The square** does not get that far. Its compiled rule is exact, but taught by answers it ends at a constant
answer, and told every tick it learns its training grid and fails one size up: 0.2–0.3 when trained at (3, 2), about
half when trained at (3, 3) with the causal neighbourhood. Its rows of products meet new situations with every extra
digit: a quarter to a third of a bigger grid's cells see a neighbourhood the training grid never showed, some of them
new even in the few inputs the exact rule reads.

**5. Wired from logic gates**, in the talk's spirit ("wire every 1 and 0 and make them learned"): every cell a little
circuit of learned two-input gates over the bits of its neighbourhood, snapped to exact Boolean gates at the end.
Taught by lessons it answers a constant. Even told every tick, it does not learn the rule in 10,000 steps: the gates
learn to pass bits along (N's digit, the borrow, the gate) but not the digit arithmetic, because which bits meet in a
gate is fixed at random, and a subtraction digit needs a few particular bits to meet.

**6. Wired by a generator: every cell connected to every other.** This arm follows the user's own intuition: connect
every cell to every other, and let one small network g produce each connection's weight from a description of the pair
(how far apart the two cells are, their kinds, how near the edges), so that the symmetry lives in that description. It
is an indirect encoding, as in HyperNEAT. The format contains the machine exactly: 339 numbers in g wire every
connection of every grid size. Trained, it lands where the bridge's free attention did. From lessons it memorizes, and
the bypass is back: its heads read the prompt's clean digits instead of computed cells, because that pull comes from the
data, and all-to-all wiring lets it act. Told every move, it rarely fits even the tiny training grid (5 of 26 runs).
When it does, its g has found the machine's simple rules ("the cell above", "the prompt cell of my column", "the top
place of the row above"), yet it fails one size up: one description had been learned where a feature never varied (at k
= 1 a SHIFT cell's up-left source is always a prompt cell). Trained where every situation occurs, the setting in which
the fixed-neighbourhood rule certified, it does not fit at all. A nearly right wiring gets little credit: one wrong
connection keeps a third of the training answers, two keep none. Only very near the right wiring does the gradient lead
back to it.

**7. Trained long with weight decay, or letting weights move only where the questions agree, it still memorizes.**
Two known cures for memorization were tried on the lessons-only grid, far longer than it takes to fit (50,000 steps at
(3, 2, 1), 20,000 at (3, 3, 2), 4 seeds, no early stop): *trimming*, weight decay over a long run, the setting in which
networks that had memorized modular arithmetic later found the general rule ("grokking"); and *agreement*, letting a
weight move only where groups of questions (or of grid columns) push it the same way, or speeding up the weights
whose gradient keeps its sign. None certifies, none fits training on snapped digits, and one size up every run stays at
chance from start to end: no late jump. Two measurements say why, for every arm:
- decay keeps the weights small, and with small weights the cells pass blurred digits; a blurred code fits the training
  answers (at (3, 2, 1), a median of 91–99% of them at the end) without being the machine. The machine's digits must
  stay sharp tick after tick, which takes large weights: the exact rule needs about 37 in norm at (3, 2, 1), and decay
  holds the runs there at 6–19. In grokking the general rule was the smaller one; here it is the larger one, so decay
  pushes away from it;
- started at the machine's own rule, every arm loses it within a few hundred steps, the plain control included. The
  lessons, which only see blurred digits, are met as well by blurred neighbours of the rule, and Adam, the optimizer
  under every arm, keeps taking full-size steps even where every lesson is already met. A rule the training cannot
  hold is not one it can find; item 8 tries optimizers that can hold it.

Agreement does not help either: the AND-mask lets a few percent of the gradient through, and delta-bar-delta ends
with most weights near its slowest speed. Neither builds the copies: N's digit, the old register and the final borrow
never reach the cells that read them.

**8. An optimizer that can rest at the exact rule does not change the verdict.** Plain SGD, and Adam with a larger
eps, stay put where no lesson pulls, unlike Adam at its default eps. They were tried in §9's lessons-only setting at
(3, 2, 1), 4 seeds, with and without weight decay:
- **from random they memorize as before.** One size up every run stays at chance as far as it ran (13 of the 24 were
  cut off short of their 50,000 steps when Modal's spend limit stopped them; finishing them can change end-of-run
  numbers, not this); plain SGD holds the steadiest soft fit of any arm so far, and it is a blurred code (a third of
  the answers on snapped digits);
- **started at the rule, they keep it only where the lessons are met exactly**, so that nothing moves them; weight
  decay leads them out of there. Where the lessons still pull, plain SGD at a small learning rate keeps every training
  answer, but not the rule: one size up it slips (to 0.70 of the answers at (3, 2, 2) after 10,000 steps), and
  99.7–99.9% of its one-tick errors are in situations the training grid never shows. The lessons hold the rule only
  where the training grid goes;
- **at (3, 3, 2)**, where almost every situation occurs, plain SGD holds the rule, exactly or nearly, until within 100
  steps it loses all of it, in every seed (at step 700–8,900).

**Against the bridge.** The bridge's attention fell for the bypass (reading the prompt's clean digits instead of the
computed ones) and learned its wiring only from menus of about 10 options, or by trying whole options and keeping
those that pass. The NCA has no bypass, since no cell near the answer can see the prompt, and no menu: its wiring is
plumbing, a chain of learned copies. From lessons it lands where the bridge's loosest networks did, memorizing; told
every move, it learns the exact machine.

**What it says about "smaller units".**
- **The smaller unit holds the algorithm, and the loops emerge.** One identical cell with a 56-unit rule is the whole
  long division; the same rule runs every size, and the computation is a stable resting state of the grid. The first
  and third parts of the pattern hold for running the algorithm.
- **The unfamiliar level below is plumbing.** In the bridge a wire was one choice from a menu. In the NCA, getting N's
  digit to the cell that needs it is a chain of copies through every cell in between, and the old register and the
  final borrow travel the same way. Every computing part has a local check in this project (the subtraction laws work
  here too); a copy has none, and the answers at the far end are too weak a signal to grow the chains.
- **The air is still sucked out, inside the cell.** The cells are small, but each cell's rule is a dense function of
  its whole neighbourhood, so it entangles what it needs with what happens to be next to it. That is why it fails
  where the surroundings are new, and why it extrapolates once training has shown it every surrounding. Even the
  exact rule, trained further on the lessons, drifts exactly there (§10). Making the
  rule sparse with a penalty does not pick the right inputs (the training grid cannot say which of two always-together
  features matters), and two-input logic gates on random wires do not find them at all.
- **A concise description of all-to-all wiring is not easier to find than the wiring.** The generator does say the
  whole wiring in a few hundred numbers, as the user expected, and the same g wires any grid. But gradient descent on
  g meets the bridge's traps: the clean prompt pulls the heads to it, and many descriptions agree with the right one
  on a small training grid. Symmetry in the description helps once the description is right; it does not help find
  it.
- **The usual cures for memorization assume the general rule is the small one.** Weight decay favours small weights,
  agreement favours what every question shares. The machine's rule is shared by every question, but it is not small:
  its digits must stay sharp through every tick, which takes large weights, while a blurred code that fits the
  training answers is cheap. So decay favours the memorizer here, and agreement alone only slows the fit.
- **Steering works when it is as uniform and local as the units.** The trace is exactly the pattern's "clean, uniform
  way of directing the units": the same signal at every cell and every tick. With it the grid learns quickly and
  exactly. The task does not supply it; where such a per-cell signal could come from without the answers spelling it
  out, for the copies especially, is the open question this leaves.

## Contents

1. The grid and the compiled rule
2. Training from random: the rule, the states, the lessons
3. From random, taught by lessons: it memorizes
4. Told every tick: it learns the training grid, and growing the grid is the problem
5. The locality ladder
6. The square
7. The bit-level arm
8. Wiring from a generator: every cell connected to every other
9. Trained long, with trimming or agreement: lessons only
10. An optimizer that can rest at the rule: lessons only
11. The traps, and the project's fixes, against the bridge
12. Caveats
13. Reproduce

## 1. The grid and the compiled rule

`grid.py`, `model.py`, `compile.py`. The task is the bridge's tiny division core ([BRIDGE.md](../bridge/BRIDGE.md) §1):
base B, moduli N with w digits (top digit nonzero), dividends v < N · B^k (k quotient digits), answer v mod N. Digits
are written lowest first.

**The worksheet.** A column is one place of a long register of k + w + 1 places, lowest at the left. Places 0 .. k − 1
hold the dividend's low digits that are still to be brought in (the queue); places k .. k + w are the machine's
register. The rows, top to bottom: the prompt, then for each of the k steps a SHIFT row and B − 1 links, each a SUB
row and a SEL row. That is the bridge's machine (H1's commit-gate chain): SHIFT moves the register up one place, so the
next dividend digit enters from the queue; SUB subtracts N, one place per cell; SEL keeps the old register if the
subtraction borrowed. The answer is the last row's places k .. k + w − 1.

Every row is uniform across its places. Subtracting N's digit 0 from a queue place with no borrow in leaves it as it
was, so the queue rides along unchanged and enters the register one place per SHIFT, as in a shift register. (The
bridge instead wrote each digit to bring in as a fresh input cell at the row that needs it.) A ring of OUT cells
surrounds the grid; a cell sees its neighbours' kinds, so it can tell where the edge is.

**A cell's state** is a few discrete channels: d, a digit; n, N's digit at this place, conveyed down from the prompt;
b, a borrow; m, a memo of the minuend; g, a gate bit. The prompt row writes v into d and N into n. Scratch cells
start blank.

**The compiled rule**, relative to a cell (UL, U, L, R are its up-left, up, left and right neighbours):

| kind | rule |
|---|---|
| SHIFT | d = UL.d (the register moved up one place; at the left edge a 0 comes in) |
| SUB | d, b = U.d − U.n − L.b, a digit and a borrow; m = U.d (one place of the link's ripple, left to right) |
| SEL | g = (R is the edge) ? U.b : R.g (the link's final borrow, passed right to left); d = g ? U.m : U.d |
| every scratch kind | n = U.n (N's digits conveyed straight down) |

Every read is inside the 3 × 3 neighbourhood. The bridge's long-range ports became three channels: N's row became
the conveyor n, "the top place of the row above" became the gate chain g, and "two rows up" became the memo m.

**The example**, B = 3, N = 5 = `2 1`, v = 14 = `2 1 1` (lowest first). Settled, the grid's digits are:

```
                 place 0 1 2 3
PROMPT   d (v)         2 1 1 0
         n (N)         0 2 1 0
SHIFT    d             0 2 1 1      register (places 1-3) = 14
SUB      d             0 0 0 1      14 - 5 = 9, no borrow
SEL      d             0 0 0 1      keep 9
SUB      d             0 1 1 0      9 - 5 = 4, no final borrow
SEL      d             0 1 1 0      answer: places 1-2 = 4
```

The second SUB row's borrow channel is `0 1 1 0`: places 1 and 2 borrow, and the top place does not, so the gate stays
0 and the SEL row takes the new value. From random scratch cells the answer is right after 13 ticks, and every cell of
the grid after 17.

**As one network.** The rule is an MLP every cell runs: its first layer is a 3 × 3 convolution over the
neighbourhood's kinds and channel values (one-hot), its second a 1 × 1 one, the shape of Mordvintsev et al.'s growing
NCA with the raw neighbourhood as the perception. Compiled, each hidden unit is one AND clause (the cell's kind and up
to three neighbour values, or "R is the edge"), and its output weights are β = 20 on the values the clause sets: 56
units at base 3. The prompt is re-injected after every tick, or written only at tick 0, in which case the rule holds
it (one more clause per prompt channel).

**The square.** x² mod N with x < N, at (B, w). The prompt writes x into d and N into n, on a long register of 2w + 1
places, and k = w:
- a SPLIT row copies x into two registers, a multiplicand a and a multiplier q;
- w MUL rows do schoolbook multiplication. Row i adds x · x_i · B^i into d: the multiplicand moves up one place per
  row (a reads UL), and the multiplier moves down one (q reads UR), so the multiplier digit x_i always arrives at place
  0. From there a channel s broadcasts it to the right, and a carry c ripples left to right: d, c = U.d + U.a · s + L.c;
- then the division rows reduce x², as above.

For N = 5, x = 4 = `1 1`: MUL(0) leaves 4 = `1 1`, MUL(1) leaves 16 = `1 2 1`, and the division brings 16 down to 1.
The compiled square rule has 242 hidden units at base 3.

**Longer reaches** (`model.Cfg.extra`), for the locality ladder of §5:
- "ncol": each cell also reads the prompt cell of its column, N's digit, so nothing conveys N (47 units);
- "ports": also the cell two rows up and the top place of the row above, the bridge's long-range ports. The select
  then reads the old register and the final borrow directly, needing neither memo nor gate chain, and a cell is the
  bridge's digit and borrow (37 units).

**The check** (record `records/compile.json`). Every question of each size, or a random 20,000 for the biggest:
- the hard path (argmax states) must hold the machine's value (`grid.machine_reference`, computed on integer digits,
  not by any network) in every channel of every cell: after the settling depth from blank scratch cells, from random
  ones, and still after twice as many ticks;
- the soft path (softmax states, the network as trained) must put almost all probability on the right answer;
- the same with the prompt written only at tick 0.

<!-- COMPILE -->
| grid | questions | cells | ticks: every cell / the answer | hidden units | exact | prompt at tick 0 only | + N's column (units) | + the bridge's ports (units) | smallest soft p(answer) |
|---|---|---|---|---|---|---|---|---|---|
| v mod N (2, 3, 3) | 176 | 70 | 45 / 23 | 36 | yes | yes | · | · | 0.999999998 |
| v mod N (2, 4, 4) | 1,472 | 117 | 76 / 32 | 36 | yes | yes | yes (30) | yes (22) | 0.999999998 |
| v mod N (3, 2, 1) | 99 | 24 | 17 / 13 | 56 | yes | yes | yes (47) | yes (37) | 0.999999996 |
| v mod N (3, 2, 2) | 297 | 55 | 42 / 23 | 56 | yes | yes | · | · | 0.999999996 |
| v mod N (3, 2, 3) | 891 | 96 | 75 / 33 | 56 | yes | yes | · | · | 0.999999996 |
| v mod N (3, 2, 4) | 2,673 | 147 | 116 / 45 | 56 | yes | yes | · | · | 0.999999996 |
| v mod N (3, 3, 2) | 2,835 | 66 | 50 / 28 | 56 | yes | yes | yes (47) | yes (37) | 0.999999996 |
| v mod N (3, 3, 3) | 8,505 | 112 | 87 / 39 | 56 | yes | yes | · | · | 0.999999996 |
| v mod N (3, 4, 3) | 20,000 (sample) | 128 | 99 / 41 | 56 | yes | yes | · | · | 0.999999996 |
| v mod N (4, 2, 2) | 1,824 | 75 | 62 / 32 | 80 | yes | yes | yes (68) | yes (56) | 0.999999994 |
| v mod N (4, 2, 3) | 7,296 | 132 | 111 / 48 | 80 | yes | yes | · | · | 0.999999994 |
| v mod N (4, 3, 2) | 30,336 | 90 | 74 / 40 | 80 | yes | yes | · | · | 0.999999994 |
| v mod N (5, 2, 2) | 7,250 | 95 | 82 / 44 | 108 | yes | yes | · | · | 0.999999992 |
| v mod N (5, 3, 2) | 20,000 (sample) | 114 | 98 / 49 | 108 | yes | yes | · | · | 0.999999992 |
| x² mod N (2, 3) | 22 | 98 | 51 / 25 | 85 | yes | · | · | · | 0.999999998 |
| x² mod N (2, 4) | 92 | 162 | 84 / 39 | 85 | yes | · | yes (75) | yes (67) | 0.999999998 |
| x² mod N (2, 5) | 376 | 242 | 125 / 51 | 85 | yes | · | · | · | 0.999999998 |
| x² mod N (3, 2) | 33 | 70 | 46 / 27 | 242 | yes | · | yes (227) | yes (217) | 0.999999996 |
| x² mod N (3, 3) | 315 | 140 | 93 / 43 | 242 | yes | · | yes (227) | yes (217) | 0.999999996 |
| x² mod N (3, 4) | 2,889 | 234 | 156 / 59 | 242 | yes | · | · | · | 0.999999996 |
| x² mod N (4, 2) | 114 | 90 | 66 / 35 | 623 | yes | · | · | · | 0.999999994 |
| x² mod N (4, 3) | 1,896 | 182 | 135 / 61 | 623 | yes | · | · | · | 0.999999994 |
| x² mod N (5, 2) | 290 | 110 | 86 / 43 | 1396 | yes | · | · | · | 0.999999992 |
<!-- /COMPILE -->

## 2. Training from random: the rule, the states, the lessons

`model.py`, `train.py`, `suites.py`. The rule is the generic 3 × 3 MLP of §1 with 128 hidden units, started from
PyTorch-style random weights; every run below starts there. The knobs:
- **the neighbourhood**: "noself" (the default) hides the cell's own state from its rule, though not its kind; "3x3"
  shows it; "causal" also hides the states of the row below, so information only flows down the grid and along rows.
  The compiled rule reads neither a cell's own state nor the row below, so all three contain it;
- **the states**: "soft" (softmax distributions passed on), or "ste" (straight-through: the forward pass snaps every
  channel to its argmax, the backward pass uses the softmax's gradient). Measurement is always on the hard path:
  argmax states, the compiled depth's number of ticks, blank scratch cells;
- **the format**: "compiled" (the channels d, n, b, m, g, with only d, n and b given a meaning by any lesson), "free"
  (d and n written by the prompt, three hidden channels of 2, 3 and 2 values that no lesson reads), or, with the
  bridge's ports, "dnb" (d, n and b only: the bridge's digit and borrow);
- **the reach**: the 3 × 3 alone, or with N's column, or with the bridge's ports (§1).

Training uses every question of the size each step (full batch; batches of 256 at (3, 3, 2)), Adam at lr 0.003
(0.03–0.05 for the logic gates), a tick count drawn from depth to 1.5 × depth each step, at most 3,000 steps (10,000
for the logic gates); a run stops 100 steps (in the later suites 300) after its hard network first gets every
training answer right, or after 1,500 steps without progress. It is **certified**, as in the bridge, if it is also
exact on every question of (B, w, k + 1) and (B, w + 1, k + 1) (up to 20,000 sampled; 2,000–5,000 in the suites
trained at (3, 2, 2) and (3, 3, 2), and every question in `cover2`).

**The lessons**, in their most generic forms, by what each needs:

| lesson | what it says | needs |
|---|---|---|
| ans | the answers | the prompt and the answer |
| xmodn | x mod N = x on each question's own register start R0 < N, and N mod N = 0 | the data's own numbers |
| step | f(N, v) = f(N, B · f(N, v // B) + v₀), the network's own soft answer fed back (k ≥ 2) | the same |
| link | the grid of one link (k = 1, one SUB and one SEL), which the same rule runs: x = R0 stays x, x = N becomes 0; and, in the compiled format, the SUB row's final borrow is [x < N] | the grid's kinds |
| pref, colall, rowall, bij | ANCHORS.md's lessons on the subtraction cell: the top-part order facts, the two one-digit order laws, and every row and column of the table using each digit once | a recognizable subtraction cell: the rule is queried at a SUB cell whose up neighbour holds d = a and n = m and whose left neighbour holds b = c, the rest of the neighbourhood blank |
| indep | the answer must not depend on the scratch cells' start: two runs from random scratch states agree | nothing |
| sheet | the worksheet's visible digits: d at every scratch cell is the machine's | the machine's worksheet (labels) |
| trace | along the compiled NCA's own run (from random and from blank scratch cells, alternately), every tick of the learned rule gives the compiled rule's next state, in every channel of every cell (`trace_chans`: some channels only; `trace_ticks`: a few sampled ticks per step) | the compiled rule itself: an upper bound, not a lesson |

G is ans + xmodn; GL adds link; GCL adds the cell's lessons. The step law (k ≥ 2) is implemented, but none of the
runs in §3–§8 turned it on: the rows trained at k = 2 say "step law off". §9 is the first to use it. The cell's
lessons fix more design than in the bridge: there, the subtraction cell's ports were fixed, so querying it fixed only
the format. Here the query
also says where the cell reads its minuend, subtrahend and borrow (U.d, U.n, L.b): it is the compiled wiring, stated as
a lesson.

## 3. From random, taught by lessons: it memorizes

`suites.py` (`explore`, `ladder2`, `ladder3`, `free2`, `sparse1`), records in `runs/<suite>.jsonl`, 2 seeds per arm.
Every arm the project's lessons can express, at the bridge's tiniest size, (3, 2, 1): 99 questions, 24 cells, 17
ticks. "fit" counts runs exact on training but wrong on a bigger size; the sub table is the rule's subtraction table
queried on the blank context (of 18 digits and 18 borrows); "answers 0" is the median share of training questions
answered 0.

<!-- LESSONS -->
| notch | lessons | certified | training, snapped / soft | bigger, snapped | sub table | answers 0 |
|---|---|---|---|---|---|---|
| noself | answers | 0/2 | 0.49 / 1.00 | 0.17 / 0.05 | 6 + 8 | 0.26 |
| noself, snapped (STE) | answers | 0/2 | 0.64 / 0.56 | 0.16 / 0.06 | 8 + 9 | 0.23 |
| 3x3 (the cell sees its own state) | answers | 0/2 | 0.35 / 0.67 | 0.16 / 0.04 | 6 + 7 | 0.32 |
| noself | GL: answers, x mod N = x, one-link law | 0/2 | 0.47 / 0.87 | 0.13 / 0.05 | 6 + 8 | 0.19 |
| noself | GL + the answer must not depend on the scratch start | 0/2 | 0.29 / 0.51 | 0.15 / 0.05 | 6 + 9 | 0.30 |
| noself | GCL: GL + the cell's subtraction lessons | 0/2 | 0.22 / 0.45 | 0.18 / 0.06 | 18 + 18 | 0.33 |
| noself, snapped (STE) | GCL | 0/2 | 0.71 / 0.66 | 0.15 / 0.05 | 18 + 18 | 0.20 |
| noself | GCL, the cell's lessons first (the rest from step 300) | 0/2 | 0.33 / 0.63 | 0.19 / 0.06 | 18 + 18 | 0.15 |
| noself, snapped (STE) | GCL, the cell's lessons first | 0/2 | 0.80 / 0.74 | 0.14 / 0.05 | 18 + 18 | 0.19 |
| noself | GCL, the cell's lessons and one link first (the rest from step 600) | 0/2 | 0.34 / 0.58 | 0.16 / 0.05 | 17 + 18 | 0.10 |
| noself, lr 0.01 | GCL | 0/2 | 0.32 / 0.41 | 0.13 / 0.05 | 17 + 18 | 0.18 |
| noself | GCL, the cell's lessons on random contexts | 0/2 | 0.45 / 0.90 | 0.16 / 0.05 | 8 + 14 | 0.21 |
| 3x3 | GCL | 0/2 | 0.37 / 0.48 | 0.18 / 0.06 | 18 + 18 | 0.23 |
| noself, random scratch starts | GCL | 0/2 | 0.36 / 0.71 | 0.15 / 0.05 | 18 + 18 | 0.17 |
| noself, one MLP block per channel | GCL | 0/2 | 0.33 / 0.51 | 0.12 / 0.05 | 18 + 18 | 0.16 |
| noself | answers + the worksheet's visible digits | 0/2 | 0.24 / 0.35 | 0.14 / 0.05 | 5 + 7 | 0.23 |
| noself, trained at (3, 2, 2) | answers | 0/2 | 0.18 / 0.18 | 0.19 / 0.05 | 6 + 9 | 0.61 |
| noself, trained at (3, 2, 2) | GCL (step law off) | 0/2 | 0.19 / 0.19 | 0.17 / 0.06 | 18 + 18 | 0.52 |
| noself, STE, trained at (3, 2, 2) | GCL (step law off) | 0/2 | 0.18 / 0.18 | 0.18 / 0.06 | 18 + 18 | 1.00 |
| noself, trained at (3, 2, 2) | answers + the worksheet's visible digits | 0/2 | 0.18 / 0.62 | 0.14 / 0.04 | 7 + 6 | 0.19 |
| noself + N's column | GCL | 0/2 | 0.65 / 0.88 | 0.15 / 0.05 | 18 + 18 | 0.20 |
| noself + the bridge's ports, digit-and-borrow cells | answers | 0/2 | 0.52 / 0.83 | 0.16 / 0.05 | 4 + 8 | 0.27 |
| noself + the bridge's ports, digit-and-borrow cells | GCL | 0/2 | 0.56 / 0.76 | 0.22 / 0.07 | 18 + 18 | 0.24 |
| the same, STE | GCL | 0/2 | 0.87 / 0.86 | 0.19 / 0.06 | 18 + 18 | 0.19 |
| noself + the bridge's ports, digit-and-borrow cells | GCL, the cell's lessons first | 0/2 | 0.55 / 0.85 | 0.14 / 0.05 | 18 + 18 | 0.18 |
| the same, trained at (3, 2, 2) | GCL (step law off) | 0/2 | 0.25 / 0.34 | 0.18 / 0.06 | 18 + 18 | 0.24 |
| causal | answers | 0/2 | 0.41 / 1.00 | 0.12 / 0.04 | 5 + 9 | 0.15 |
| causal | GCL | 0/2 | 0.31 / 0.71 | 0.17 / 0.05 | 18 + 18 | 0.17 |
| causal, STE | GCL | 0/2 | 0.79 / 0.80 | 0.19 / 0.05 | 18 + 18 | 0.19 |
| causal, trained at (3, 2, 2) | GCL (step law off) | 0/2 | 0.15 / 0.25 | 0.16 / 0.05 | 18 + 18 | 0.51 |
| causal, trained at (3, 3, 2) | GCL (step law off) | 0/2 | 0.07 / 0.18 | 0.04 / 0.01 | 18 + 18 | 0.05 |
| causal, L1 1.0 on the input weights | GCL | 0/2 | 0.49 / 0.93 | 0.17 / 0.06 | 18 + 18 | 0.16 |
| causal, STE, L1 1.0 | GCL | 0/2 | 0.84 / 0.84 | 0.18 / 0.06 | 18 + 18 | 0.19 |
| free format, noself | answers | 0/2 | 0.42 / 1.00 | 0.13 / 0.05 | · | 0.16 |
| free format, noself | GL | 0/3 | 0.26 / 0.32 | 0.18 / 0.05 | · | 0.14 |
| free format, noself, STE | GL | 0/4 | 0.87 / 0.83 | 0.14 / 0.05 | · | 0.19 |
| free format with 6 hidden channels | GL | 0/2 | 0.32 / 0.65 | 0.16 / 0.05 | · | 0.16 |
| free format, noself | GL + the answer must not depend on the scratch start | 0/2 | 0.42 / 0.82 | 0.12 / 0.05 | · | 0.18 |
| free format, noself | answers + the worksheet's visible digits | 0/2 | 0.24 / 0.35 | 0.14 / 0.05 | · | 0.23 |
| free format, noself, STE | answers + the worksheet's visible digits | 0/2 | 0.98 / 0.94 | 0.16 / 0.05 | · | 0.19 |
| free format, causal, trained at (3, 2, 2) | answers + the worksheet's visible digits | 0/2 | 0.19 / 0.62 | 0.15 / 0.04 | · | 0.16 |
<!-- /LESSONS -->

What the table says:
- **No run certifies, and almost none fits.** 3 runs ever get every training question right on the snapped
  path, all with snapped states, and lose it again. One size up every run is within a few points of "always answer
  0" (0.18 at (3, 2, 2), 0.06 at (3, 3, 2); the best run gets 0.24 and 0.08).
- **Soft states memorize through an analog code.** On answers alone the soft network fits all 99 training questions
  in 7 of 10 runs (soft share 1.00) while the snapped one gets a third to a half: blurred digits pass between cells a
  code that exact digits cannot carry. The bridge's loose notches did the same ([BRIDGE.md](../bridge/BRIDGE.md) §3,
  "softcert").
- **Snapped states (straight-through) memorize in hard form.** STE closes the gap between the two paths and raises the
  snapped training share to 0.69–0.98 (one run 0.32); some runs fit every training question at some step, but one
  size up they are at chance. The digits are now exact, and they hold a lookup arrangement instead of the algorithm.
- **The cell's lessons work on the cell and not on the grid.** With them the queried subtraction table comes out
  exact (18 + 18) in nearly every run, as in the bridge. The grid around it never forms: N's conveyor, the memo, the
  gate chain and the shift are plumbing, and no lesson checks them locally.
- **Neither staging nor the sub-grid first helps.** The cell's lessons first ("tables first", the bridge's fix for its
  menu), or the cell's lessons and the one-link grid first, end where GCL does.
- **Less locality does not help either**: N's column read directly, or the bridge's three long-range ports with
  digit-and-borrow cells. With every source the bridge's machine reads in plain view, the rule still has to learn
  which of its 12 sources each kind reads, and it memorizes instead.
- **Neither do the worksheet's visible digits.** Labelling d at every cell of every row with the machine's digit, the
  student's written work, does not even fit training with soft states (with snapped states it fits by a lookup, like
  every snapped arm): the hidden channels that carry N, the memo and the gate get no label, and they do not form.
- **The free format** (hidden channels with learned meanings) behaves the same. Probed against the machine's channels
  (`relabel`: how well a lookup of a hidden channel's value predicts N's digit, the borrow, the memo or the gate,
  above the majority value), its hidden channels carry at most a shadow of them: a lift of at most 0.4, once 0.65.

## 4. Told every tick: it learns the training grid, and growing the grid is the problem

`suites.py` (`explore`, `ladder2`, `ladder3`, `causal2`, `sparse1`, `partial1`, `selfcopy2`, `diag1`,
`cover1`–`cover3`). To separate what the format can learn from what the lessons can teach, the trace lesson tells the
rule its every move: along the compiled NCA's own run (from random and from blank scratch cells, alternately), each tick
of the learned rule must give the compiled rule's next state in every channel of every cell. That is far more than the
project's lessons: traces as supervision, which RELAXED.md found to certify H1 when every trace token is a local
operation.

<!-- TRACE -->
| neighbourhood | trained at | certified | training | k + 1 | k + 2 | w + 1, k + 1 | w + 2, k + 2 |
|---|---|---|---|---|---|---|---|
| 3x3 | (3, 2, 1), only the settled grid told | 0/2 | 0/2 exact | 0.182 | · | 0.057 | · |
| noself | (3, 2, 1), only the settled grid told | 0/2 | 0/2 exact | 0.184 | · | 0.057 | · |
| noself | (3, 2, 1) | 0/2 | 2/2 exact (step 550) | 0.655 | 0.348 | 0.408 | 0.081 |
| 3x3 | (3, 2, 1) | 0/2 | 2/2 exact (step 787) | 0.483 | 0.269 | 0.314 | 0.063 |
| noself + the bridge's ports | (3, 2, 1) | 0/2 | 2/2 exact (step 425) | 0.296 | 0.169 | 0.129 | 0.020 |
| noself | (3, 2, 2) | 0/2 | 2/2 exact (step 575) | 1.000 | 1.000 | 0.877 | · |
| noself | (3, 3, 2), batches of 256 | 0/2 | 2/2 exact (step 700) | 1.000 | 1.000 | 0.992 (min 0.989) | · |
| causal | (3, 2, 1) | 0/6 | 6/6 exact (step 337) | 0.907 | 0.731 | 0.731 | 0.363 |
| causal | (3, 2, 2) | 0/2 | 2/2 exact (step 275) | 1.000 | 1.000 | 0.774 | · |
| causal | (3, 3, 2), batches of 256 | 6/6 (475) | 6/6 exact (step 475) | 1.000 | 1.000 | 1.000 | 1.000 (min 0.995) |
| causal | (3, 3, 2), batches of 256, every question checked | 4/8 (437) | 8/8 exact (step 462) | 1.000 | 1.000 | 1.000 (min 0.989) | · |
| noself, L1 0.3 on the input weights | (3, 2, 1) | 0/2 | 2/2 exact (step 537) | 0.682 | 0.481 | 0.519 | 0.227 |
| noself, L1 1.0 on the input weights | (3, 2, 1) | 0/2 | 2/2 exact (step 625) | 0.785 | 0.550 | 0.632 | 0.259 |
| noself, L1 3.0 on the input weights | (3, 2, 1) | 0/2 | 2/2 exact (step 937) | 0.650 | 0.469 | 0.496 | 0.215 |
| causal, L1 0.3 on the input weights | (3, 2, 1) | 0/2 | 2/2 exact (step 300) | 0.921 | 0.834 | 0.815 | 0.552 |
| causal, L1 1.0 on the input weights | (3, 2, 1) | 0/2 | 2/2 exact (step 337) | 0.835 | 0.597 | 0.723 | 0.348 |
| causal, L1 3.0 on the input weights | (3, 2, 1) | 0/2 | 2/2 exact (step 412) | 0.929 (min 0.923) | 0.705 | 0.806 | 0.421 |
| causal | (3, 2, 1), only d told | 0/2 | 0/2 exact | 0.098 | · | 0.038 | · |
| causal | (3, 2, 1), only d, b told | 0/2 | 0/2 exact | 0.121 | · | 0.036 | · |
| causal | (3, 2, 1), only d, b, n told | 0/2 | 0/2 exact | 0.086 | · | 0.031 | · |
| causal | (3, 2, 1), only d, b, n, m told | 0/2 | 0/2 exact | 0.215 | · | 0.123 | · |
<!-- /TRACE -->

**The training grid is learned in a few hundred steps.** Every MLP run told every channel fits it, in 250–850 steps
(up to 975 with the strongest L1 penalty), every cell and every channel the machine's, and the result is a stable
loop: exact still at 2× and 4× the ticks. Asking only that the machine's settled grid be a resting state is not
enough (`selfcopy2`, the TRACE rows "only the settled grid told"): every rule learns it, and every run stays at the
blank start, by copying its own state where it can see it, and otherwise because the blank grid is a resting state too.
Telling only some channels (`partial1`, rows "only d, b, ... told") does not fit training either: a label one tick from
the machine's state trains only the channel it labels, so the untold ones (the gate, say) never form, and every
channel has to be told.

**Growing the grid fails, for two reasons** (`diag1`, records in `runs/diag1.jsonl`): one tick from the machine's own
state at a bigger size, which cells does the trained rule get wrong?
1. **A situation never met in training.** At (3, 2, 2) the only wrong cells are in the second step's SHIFT row: the
   first SHIFT row ever to sit under a SEL row. At k = 1 every SHIFT row sits under the prompt. Its places are right
   on 52–91% of questions; every other row is exact.
2. **A second resting state.** On the other seed, every cell of (3, 2, 2) is right one tick from the machine's state,
   yet the free run from blank cells settles elsewhere (the first step's places at 0.97, the second's at 0.8–0.9). The
   learned rule also reads the row below, which the compiled one never does; so rows feed back into each other, and
   the dynamics have more than one self-consistent grid.

**Removing the feedback fixes the second.** With the causal neighbourhood (the row below hidden), one size up goes
from 0.57–0.74 to 0.69–0.99 at (3, 2, 2) (median 0.91 over 6 seeds), and from 0.32–0.50 to 0.50–0.82 at (3, 3, 2)
(median 0.73).

**The first is about how many situations a bigger grid poses** (`situations.py`, `records/situations.json`), counted
on the machine's settled states:

<!-- SITUATIONS -->
| trained at | measured at | cells meeting a new 3x3 kind pattern | a new full neighbourhood | new inputs of the compiled rule |
|---|---|---|---|---|
| (3, 2, 1) | (3, 2, 2) | 20.0% | 33.1% | 0.0% |
| (3, 2, 1) | (3, 3, 2) | 20.0% | 52.4% | 0.8% |
| (3, 2, 1) | (3, 3, 3) | 26.7% | 64.2% | 0.6% |
| (3, 2, 1) | (3, 4, 3) | 26.7% | 69.5% | 1.4% |
| (3, 2, 2) | (3, 3, 3) | 0.0% | 24.0% | 0.6% |
| (3, 2, 2) | (3, 4, 3) | 0.0% | 34.2% | 1.4% |
| (3, 3, 2) | (3, 3, 3) | 0.0% | 0.5% | 0.0% |
| (3, 3, 2) | (3, 4, 3) | 0.0% | 8.5% | 0.0% |
| the square at (3, 3) | the square at (3, 4) | 0.0% | 24.2% | 1.3% |
| the square at (3, 3) | the square at (3, 5) | 0.0% | 32.7% | 2.1% |
<!-- /SITUATIONS -->

A third to two thirds of a bigger grid's cells see a neighbourhood that never occurs at (3, 2, 1), counting
everything the rule reads. Counting only what the compiled rule reads, almost none do. So a rule that ignored the rest
would meet nothing new; the learned rule, a dense function of its whole neighbourhood, does not ignore it. An L1
penalty on its input weights (`sparse1`) shrinks what it reads (to 46 of 110 inputs at the strongest), but does not
reliably help: the training grid cannot say which of two always-together features (a SHIFT cell's own kind, or the
prompt above it) is the one to keep.

**So train where every situation occurs.** At (3, 2, 2) every row-to-row situation occurs (a SHIFT row under a SEL
row included), and the trace-taught rule is then exact with any number of quotient digits checked: (3, 2, 3) and
(3, 2, 4), on both neighbourhoods. A new width still breaks it (0.75–0.89 at (3, 3, 3)): two inner places side by
side never occur with w = 2. At (3, 3, 2) almost every situation of every bigger grid occurs (the table above), and
there:
- **with the causal neighbourhood the rule certifies** (`ladder3`, `cover1`, `cover2`, `cover3`). Checked the
  bridge's way, on samples of 2,000–3,000 questions of each bigger size, all 6 seeds certify; they also stay exact at
  2× and 4× the ticks, at (3, 3, 4) and at (3, 4, 4), and 3 of the 4 measured there are exact at (3, 5, 4), two
  digits of N beyond training (the fourth 0.995). Checked on every question, the 8,505 of (3, 3, 3) and the 78,003 of
  (3, 4, 3), 4 of 8 seeds certify. All 8 are exact at the training width: on every question of (3, 3, 3), and on
  (3, 3, 4) (every one of its 25,515 questions for the six rules verified below, a sample of 5,000 for the other two);
  the other four miss 0.03–1.1% of (3, 4, 3);
- with the row below visible, it comes close (1.00 at (3, 3, 3), 0.989–0.995 at (3, 4, 3)) and does not certify.

**Further out** (`verify.py`, `records/verify.json`), from the weights `cover2` and `cover3` kept: every question where
marked, else a sample of 3,000; scratch cells blank or random; the compiled depth's ticks, or 4×.

<!-- VERIFY -->
| size, start, ticks | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | seed 11 |
|---|---|---|---|---|---|---|
| (3, 3, 2), blank, 1× ticks | all 2,835 | all 2,835 | all 2,835 | all 2,835 | all 2,835 | all 2,835 |
| (3, 3, 3), blank, 1× ticks | all 8,505 | all 8,505 | all 8,505 | all 8,505 | all 8,505 | all 8,505 |
| (3, 3, 3), random, 1× ticks | all 8,505 | all 8,505 | all 8,505 | all 8,505 | all 8,505 | all 8,505 |
| (3, 3, 3), blank, 4× ticks | all 8,505 | all 8,505 | all 8,505 | all 8,505 | all 8,505 | all 8,505 |
| (3, 4, 3), blank, 1× ticks | all 3,000 sampled | all 3,000 sampled | 2,986 of 3,000 sampled | all 3,000 sampled | all 3,000 sampled | all 3,000 sampled |
| (3, 4, 3), random, 1× ticks | all 3,000 sampled | all 3,000 sampled | 2,987 of 3,000 sampled | all 3,000 sampled | all 3,000 sampled | all 3,000 sampled |
| (3, 4, 3), blank, 4× ticks | all 3,000 sampled | 2,999 of 3,000 sampled | 2,986 of 3,000 sampled | all 3,000 sampled | all 3,000 sampled | all 3,000 sampled |
| (3, 3, 4), blank, 1× ticks | all 25,515 | all 25,515 | all 25,515 | all 25,515 | all 25,515 | all 25,515 |
| (3, 4, 4), blank, 1× ticks | all 3,000 sampled | 2,999 of 3,000 sampled | 2,976 of 3,000 sampled | all 3,000 sampled | all 3,000 sampled | all 3,000 sampled |
| (3, 4, 4), random, 1× ticks | all 3,000 sampled | 2,999 of 3,000 sampled | 2,975 of 3,000 sampled | all 3,000 sampled | all 3,000 sampled | all 3,000 sampled |
| (3, 4, 4), blank, 4× ticks | all 3,000 sampled | 2,999 of 3,000 sampled | 2,976 of 3,000 sampled | all 3,000 sampled | all 3,000 sampled | all 3,000 sampled |
| (3, 5, 4), blank, 1× ticks | 2,998 of 3,000 sampled | 2,988 of 3,000 sampled | 2,934 of 3,000 sampled | all 3,000 sampled | 2,996 of 3,000 sampled | all 3,000 sampled |
| (3, 5, 5), blank, 1× ticks | 2,995 of 3,000 sampled | 2,986 of 3,000 sampled | 2,933 of 3,000 sampled | all 3,000 sampled | 2,991 of 3,000 sampled | all 3,000 sampled |
<!-- /VERIFY -->

Every one of the six is exact on every question at its training width, w = 3, for every number of quotient digits
checked (2, 3 and 4), from blank or random scratch cells and at 4× the ticks. Wider registers are where they differ:
seeds 9 and 11 are exact on every sample out to (3, 5, 5), two more digits of N and three more quotient digits than
they were trained on; seeds 6 and 10 are exact through (3, 4, 4) and miss 0.1–0.3% at w = 5; seeds 7 and 8, which
already missed at (3, 4, 3), miss up to 2% further out.

**The misses on every question are mostly the second resting state again, now inside a row** (`where.py`,
`records/where_*.json`). They fall on a handful of moduli whose digit patterns no three-digit modulus has. Seed 7 gets
21 of the 78,003 questions of (3, 4, 3) wrong, every one with N = 80 = `2 2 2 2`, the one four-digit modulus whose
digits are all 2. Seeds 8, 12 and 13 get 459, 865 and 631 wrong, every one with N = 28, 29, 55 or 56, the four-digit
moduli with two zeros side by side inside them (`1 0 0 1`, `2 0 0 1`, `1 0 0 2`, `2 0 0 2`). For seeds 7, 8 and 13
the rule is right at every cell of every one of these questions one tick from the machine's own state, and started
from random scratch cells instead of blank ones some come out right (19 of 21; 37 of 459; 39 of 631). So the
machine's grid is a resting state of the learned rule, and from blank cells the run settles into another one. Hiding
the row below removed the feedback between rows; within a row the learned rule still reads both of its neighbours,
which the compiled rule never does for any one kind (a SUB cell reads only its left neighbour, a SEL cell only its
right), and an unfamiliar pattern of N's digits lets a row settle wrong. Seed 12 also has cells wrong one tick from the
machine's state on those questions (286 of them): a situation never met, the first reason, as well.

## 5. The locality ladder

Is locality itself the obstacle? Each rung adds fixed long-range reads to the 3 × 3 (§1, "longer reaches"), each with
its compiled point checked exact:
- **+ N's column**: every cell reads N's digit straight from the prompt cell of its column, so nothing conveys N;
- **+ the bridge's ports**: also the cell two rows up and the top place of the row above. The cell is then the
  bridge's (a digit and a borrow), and every source the bridge's machine reads is in plain view of every cell.

Taught by lessons (rows "+ N's column" and "+ the bridge's ports" of the LESSONS table, §3), neither rung learns: the
rule still has to find, among a dozen sources, which ones each kind reads, and it memorizes first, like the bridge's
free attention. Told by the trace, the ports make extrapolation worse, not better (TRACE table, §4: 0.25–0.34 one
size up against 0.57–0.74 with the 3 × 3 alone). More inputs are more situations that change with the grid's size: the
top place of the row above, for instance, is a different cell at every width.

## 6. The square

`suites.py` (`square1`, `causal2`, `partial1`). x² mod N at (B, w) = (3, 2): 33 questions, 70 cells, 46 ticks (§1),
certified if also exact at (3, 3) and (3, 4). The division's lessons run on the division's own grid, with the same
rule (its kinds include the division's); the last column is that rule's share on the plain division grid of the
same base and width, with one quotient digit.

<!-- SQUARE -->
| notch | lessons | certified | training, snapped / soft | bigger, snapped | the plain division grid (B, w, 1) |
|---|---|---|---|---|---|
| noself | the square's answers | 0/2 | 0.42 / 0.42 | 0.03 / 0.01 | 0.18 |
| noself, STE | the square's answers | 0/2 | 0.42 / 0.42 | 0.03 / 0.01 | 0.18 |
| noself | the square's answers + the division's GCL on its own grid | 0/2 | 0.45 / 0.42 | 0.09 / 0.02 | 0.31 |
| noself | the same + the product x^2 as a label | 0/2 | 0.36 / 0.55 | 0.08 / 0.03 | 0.43 |
| noself | the compiled trace | 0/2 | 1.00 / 1.00 | 0.28 / 0.08 | 0.18 |
| causal | the compiled trace | 0/2 | 1.00 / 1.00 | 0.31 / 0.09 | 0.19 |
| noself + the bridge's ports | the compiled trace | 0/2 | 1.00 / 1.00 | 0.20 / 0.08 | 0.18 |
| causal, trained at (3, 3), batches of 64 | the compiled trace (12 ticks a step) | 0/2 | 1.00 / 0.99 | 0.49 / 0.19 | 0.10 |
<!-- /SQUARE -->

What the table says:
- **The answers alone give a constant, or nearly.** Every square run taught by answers, with or without the
  division's lessons on its own grid, ends at 0.30–0.48 of training, and most at 0.42: exactly the 14 of 33
  questions whose answer is 1, the most common answer. Taught by its lessons alongside, the division part reaches
  0.25–0.60 of the plain division grid, and nothing one size up.
- **Told every tick, the square's grid is learned**, every cell of every row, MUL and division alike, in about 500
  steps, and is wrong one size up (0.19–0.33 at (3, 3)). Its rule cannot even run the plain division grid (0.18,
  "always 0"), whose first SHIFT row sits under the prompt: in the square's grid a SHIFT row sits under a MUL row.
  A worksheet's rows are a new situation for a rule trained on another worksheet's.
- **Trained where its grid shows more situations, it still does not extrapolate.** With the causal neighbourhood and the
  trace, trained at (3, 3) (315 questions, 140 cells, 93 ticks, in batches of 64 with 12 traced ticks a step; the
  setting in which the division certified), the square's grid is learned: every cell right one tick from the machine's
  state, and the training answers exact at step 950 and 1,025 in its two seeds (in one, one of 315 had slipped by the
  run's end, 300 steps later). One size up it gets about half: 0.46–0.52 at (3, 4), 0.18–0.21 at (3, 5).
  Its grid shows far fewer of a bigger grid's situations than the division's did (SITUATIONS table, §4): trained at (3,
  3), 24% of the cells of (3, 4) and 33% of (3, 5) see a neighbourhood it never saw, and 1.3–2.1% something new even
  among the few inputs the exact rule reads, against 8.5% and 0% for the division trained at (3, 3, 2). Every extra
  digit makes the multiplicand longer, the carries ripple further and the rows of products more numerous.

## 7. The bit-level arm

`logic.py`, `suites.py` (`explore`, `logic3`; `logic2` was stopped, below). In the talk's spirit, "wire every 1 and 0
and make them learned": the same grid, with each cell's rule a network of learned two-input logic gates, as in
differentiable logic gate networks (Petersen et al., 2022) and their use in cellular automata (Miotti, Niklasson,
Randazzo and Mordvintsev, 2025, who learned the Game of Life this way):
- **the input is the neighbourhood's bit planes**: every kind and every channel value of every neighbour is one bit
  (one-hot), 110 bits with the causal neighbourhood;
- **each gate** reads two outputs of the layer before, by a fixed random wiring, and holds 16 logits, one per Boolean
  function of two bits. In training it computes the softmax mixture of the functions' real-valued forms (AND = ab, OR
  = a + b − ab, XOR = a + b − 2ab, ...); gates start near "pass the first input through", the residual
  initialization that lets deep gate networks train;
- **the output**: the last layer is split into one group of gates per channel value, and a value's logit is its
  group's sum (difflogic's group sum);
- **evaluated hard**, each gate is its argmax function on exact bits: the rule is then a Boolean circuit, and the whole
  grid a circuit of circuits.

<!-- LOGIC -->
| rule | lessons | certified | training, snapped / soft | bigger, snapped | gates |
|---|---|---|---|---|---|
| 256 x 4 gates, noself | answers | 0/2 | 0.19 / 0.23 | 0.18 / 0.06 | 1128 |
| 256 x 4 gates, noself | GL | 0/2 | 0.20 / 0.29 | 0.19 / 0.06 | 1128 |
| 256 x 4 gates, noself | GCL | 0/2 | 0.18 / 0.18 | 0.16 / 0.06 | 1128 |
| 256 x 4 gates, noself | GCL, the cell's lessons first | 0/2 | 0.21 / 0.25 | 0.19 / 0.06 | 1128 |
| 256 x 4 gates, causal, 10,000 steps | trace (6 ticks a step) at (3, 2, 1) | 0/2 | 0.19 / 0.18 | 0.18 / 0.06 | 1128 |
| 512-512-512-256 gates, causal, 10,000 steps | trace (6 ticks a step) at (3, 2, 1) | 0/2 | 0.19 / 0.18 | 0.18 / 0.06 | 2000 |
| 512-512-512-256 gates, causal, 10,000 steps | trace (6 ticks a step) at (3, 2, 2), batches of 64 | 0/2 | 0.18 / 0.18 | 0.18 / 0.06 | 2000 |
<!-- /LOGIC -->

What the table says:
- **Taught by lessons, the gate rule stays at a constant answer** (0.17–0.20 of training, "always 0" one size up).
- **Told every tick, it does not learn the rule within 10,000 steps.** Both widths (1,128 and 2,000 gates) plateau
  with a trace loss of about 0.2, where the MLP rule reaches 0.001 in a few hundred steps, and stop for lack of
  progress after 5,000–9,000 steps. One tick from the machine's state their plumbing bits are mostly right (N's digit,
  the borrow, the memo and the gate each 0.84–1.0 of cells) and their digits are not (0.63–0.94; a SUB cell's digit
  0.63–0.74). It is the MLP's failure turned inside out. Every gate starts as "pass the first input through", so
  copying a bit along is what the gates do first; a subtraction digit depends on a few specific bits (three of U's
  digit, three of N's, two of the borrow, and the cell's kind), which random fixed wires rarely bring together in one
  gate, and the gates cannot move their wires. Trained at (3, 2, 2) in batches of 64, the mid-width rule stays at a
  constant answer (0.18).
- **The first suite ran out of memory**: the wide rule (1024-1024-512-256 gates) storing every gate's activations for
  all 17 traced ticks needs more than the 8 GB a container had, so `logic2` was stopped; `logic3` samples 6 ticks a
  step.

So "wire every 1 and 0" moves the problem rather than removing it: the gates learn, but which bits meet is the
wiring, and here the wiring is where the information is. Making the wires learnable too would be free attention over
bits, which the bridge found the hardest thing to learn. That is plausibly also why the Game of Life, which DiffLogic
CA learned, was within reach: it depends on the count of live neighbours, so any pair of bits a random wire picks
says something; a long-division cell depends on a few particular bits.

## 8. Wiring from a generator: every cell connected to every other

`hyper.py`, `suites.py` (`hyper1`–`hyper3`, `hyper_stab`, `hyper_diag`), records in `runs/hyper*.jsonl` and
`records/compile_hyper.json`. The user's own intuition: "every cell needs to be connected to every other cell and then
some higher level coordination would be able to concisely describe how to reap. And homogeneity/instancing/symmetry
seems more about that high-level description." That is an indirect encoding, as in HyperNEAT's CPPNs (Stanley,
D'Ambrosio and Gauci, 2009) and hypernetworks (Ha, Dai and Le, 2016), and the user's "parametrize in equation space"
applied to wiring:
- **connections**: every cell of the grid reads every cell, the OUT ring included, through 3 heads;
- **weights**: one small learned network g scores every connection from a description of the pair: the row and
  column offset (one-hot within 3, and as plain numbers), both cells' kinds, and how far each is from the left and
  right edges. Nothing in the description depends on the grid's size, so the same g wires any grid. The scores
  become weights by a softmax over the sources (heads compete, as in attention) or by a sigmoid each (no
  competition). Evaluated hard, a softmax head reads its best source and a sigmoid head every source above 0;
- **the rule**: the same kind of shared MLP as before, on the cell's kind and its three heads' weighted sums of
  kinds and states. Cells are the bridge's digit and borrow (the dnb format), and the trace is the bridge's machine.

**It contains the machine.** Compiled (`hyper.compile_ports`), g has one hidden unit per port, an AND of the receiver's
kind and the source's description, and reproduces the bridge's wiring: head A, B and C of a SHIFT cell read the cell
up-left; of a SUB cell, the cell above, N's digit in its column (the prompt cell, any distance up) and the cell to its
left; of a SEL cell, the cell two rows up, the cell above and the top place of the row above. 339 numbers in g wire
every connection of every grid, 3,456 at (3, 2, 1) and 57,915 at (2, 4, 4); the rule has 30 units at base 3. It is
exact on every cell of every question at eight sizes (bases 2–5, up to 4 digits of N and 4 quotient digits), with
either weighting, hard and soft, from blank and random scratch cells, at 1× and 2× the ticks.

<!-- HYPER -->
| wiring | lessons | certified | training, snapped / soft | bigger, snapped | ports right | bypass | sub table |
|---|---|---|---|---|---|---|---|
| softmax | answers | 0/2 | 0.74 / 1.00 | 0.24 / 0.07 | 0.00 | 0.66 | 6 + 9 |
| softmax | GCL | 0/2 | 0.46 / 1.00 | 0.19 / 0.06 | 0.00 | 0.56 | 18 + 18 |
| softmax, snapped (STE) | GCL | 0/2 | 0.42 / 1.00 | 0.19 / 0.06 | 0.00 | 0.67 | 18 + 18 |
| softmax | GCL, the cell's lessons first and g frozen (the rest from step 300) | 0/2 | 0.37 / 1.00 | 0.18 / 0.06 | 0.12 | 0.63 | 18 + 18 |
| softmax | GCL, the cell's lessons and one link first (the rest from step 600) | 0/2 | 0.51 / 0.99 | 0.16 / 0.05 | 0.00 | 0.65 | 18 + 18 |
| sigmoid | GCL | 0/2 | 0.39 / 0.99 | 0.19 / 0.05 | · | · | 18 + 18 |
| softmax, causal mask | GCL | 0/2 | 0.33 / 1.00 | 0.16 / 0.05 | 0.00 | 0.46 | 18 + 18 |
| softmax | trace at (3, 2, 1) | 0/12 (fit 1) | 0.30 / 1.00 | 0.18 / 0.06 | 0.12 | 0.10 | 6 + 9 |
| softmax, causal mask | trace at (3, 2, 1) | 0/12 (fit 4) | 0.50 / 1.00 | 0.15 / 0.05 | 0.25 | 0.12 | 6 + 9 |
| sigmoid | trace at (3, 2, 1) | 0/2 | 0.28 / 1.00 | 0.17 / 0.05 | · | · | 6 + 13 |
| softmax | trace at (3, 3, 2), batches of 256 | 0/2 | 0.11 / 0.57 | 0.09 / 0.05 | 0.00 | 0.22 | 6 + 9 |
| softmax, causal mask | trace at (3, 3, 2), batches of 256 | 0/4 | 0.15 / 0.20 | 0.11 / 0.06 | 0.00 | 0.18 | 6 + 9 |
<!-- /HYPER -->

What the table says (`hyper1`; the trace rows pool `hyper1`–`hyper3`, 26 runs at (3, 2, 1)):
- **From lessons it memorizes, and the bypass is back.** The soft network fits all 99 training questions in 12 of the
  14 lesson runs; snapped, they get 0.31–0.85 (the best snapped share of any lesson arm in this study), and one size up
  they are at chance (at best 0.34). Of the connections that should read a computed cell, 39–90% read a clean
  prompt cell instead (the bridge's free attention ended at about 37%; the fixed-neighbourhood NCA cannot do it at
  all), and the answers' first gradient at random initialization already points 7–59% of the connections at prompt
  cells (`hyper_diag`; the bridge measured 57%). So the pull comes from the data, as the bridge found, and all-to-all
  wiring lets it act. No generated wiring is the machine's. The cell's lessons make the subtraction table exact, as
  everywhere; freezing g while they work (staged), teaching one link first, a sigmoid weighting (no competition between
  sources) and the causal mask all end the same way.
- **Told every tick, it rarely even fits the training grid** (5 of 26 runs, against every one of the fixed-neighbourhood
  NCA's), and the fits extrapolate worst of all: 0.09–0.21 at (3, 2, 2), against 0.69–0.99 for the causal
  fixed-neighbourhood rule.
- **The fitted wirings are the machine's simple rules, nearly** (`describe_wiring`, on the saved weights of
  `hyper3`). Up to which head plays which role, a fitted SUB cell reads "the cell above", "the cell to my left" and
  "the prompt cell of my column" (two rows up for the first link, four for the second: g keyed on the prompt, not on an
  offset), and a SEL cell "two rows up", "the cell above" and "the top place of the row above". The exceptions are
  harmless where the training grid shows them: where N's digit is always 0 (the queue places and the top place), any
  blank source stands in for N's prompt cell. One size up, these descriptions still hold, at offsets of 7 and 9 rows
  it never saw.
- **What breaks is a description learned where a feature never varied.** At k = 1 a SHIFT cell's up-left source is
  always a prompt cell, so g learned "the up-left prompt cell"; at (3, 2, 2) the second step's SHIFT row sits under a
  SEL row, finds no prompt cell up-left, and reads the cell above instead (that row is right on 46–55% of questions
  one tick from the machine's state; every other row on 93–100%). It is §4's never-met situation, now inside g.
- **Trained where every situation occurs, it does not fit at all.** At (3, 3, 2), where the fixed-neighbourhood rule
  certified, none of the 6 generator-wired runs fits its training grid within 3,000 steps: they stop
  for lack of progress at 0.05–0.20 of the 2,835 training questions snapped. Two fit all of them soft (1.00) while
  their snapped versions get 0.17 and 0.20: the heads blend sources, an analog code in the wiring itself. In the two
  runs probed without regard to head order, the SHIFT cells read the machine's sources, the SUB cells mostly (63–100%
  of them) and the SEL cells never. One
  explanation, not tested here: on the bigger grid every head chooses among 104 sources instead of 48, and the soft
  search over all of them is the bridge's free attention again.

**Does a nearly right wiring get partial credit?** Two tests. First, the census of `hyper.port_swaps`: from the
compiled wiring, change one port's source to another of the bridge's 7 relative rules (42 wirings), or two ports
(200 drawn), or draw every port at random (200), and score each on the 99 training questions:

<!-- HYPER_SWAPS -->
| wirings | how many | right on every training answer | median share right | best share right |
|---|---|---|---|---|
| the compiled wiring | 1 | 1 | 1.00 | 1.00 |
| one port changed (to another of the 7 rules) | 42 | 0 | 0.33 | 0.67 |
| two ports changed | 200 | 0 | 0.18 | 0.37 |
| every port drawn at random | 200 | 0 | 0.18 | 0.33 |

First pull toward a prompt cell (share of cells and ports, random initializations): softmax all: 0.07–0.58 over 4 seeds;
softmax causal: 0.10–0.46 over 4 seeds.
<!-- /HYPER_SWAPS -->

Nearly right is not nearly as good. Not one of the 42 one-port wirings gets every training answer right: the
median gets a third (0.33, where "always 0" gets 0.18), and the best two thirds (the subtrahend read from two rows
up instead of from N's prompt cell). Two wrong ports are back at chance (median 0.18), like random wirings. The
answers grade a near miss a little above chance, but each wrong port costs most of the credit, and nothing in the
score says which port is wrong: it is all-or-nothing per port, as individually learned wiring was in the bridge.

Second, the stability test of BRIDGE.md §7: start at the compiled machine with its scales softened to 4, add noise of
std sigma to every parameter of g and the rule, and train 1,000 steps:

<!-- HYPER_STAB -->
| lessons | noise sigma | exact at the start, snapped / soft | exact after 1,000 steps, snapped / soft | certified | ports right at the end |
|---|---|---|---|---|---|
| answers | 0.1 | 0.31 / 0.07 | 0.41 / 0.71 | 0/2 | 0.94 |
| answers | 0.3 | 0.11 / 0.06 | 0.24 / 0.83 | 0/2 | 0.38 |
| answers | 1.0 | 0.05 / 0.05 | 0.11 / 0.17 | 0/2 | 0.00 |
| GCL | 0.1 | 0.31 / 0.07 | 0.91 / 1.00 | 0/2 | 1.00 |
| GCL | 0.3 | 0.11 / 0.06 | 0.24 / 0.82 | 0/2 | 0.44 |
| GCL | 1.0 | 0.05 / 0.05 | 0.18 / 0.19 | 0/2 | 0.00 |
<!-- /HYPER_STAB -->

Near the compiled point the gradient does lead back. From noise 0.1, which already leaves only 0.31 of training
answers right, GCL brings every port back to the machine's source and the answers to 0.91 snapped (1.00 soft); the
answers alone bring back 0.94 of the ports and 0.41 of the answers. From noise 0.3 it drifts to other wirings (0.38–0.44
of ports right), and from 1.0 nothing is recovered. The basin around the generated machine is narrow: the bridge's
7-option menu, on the answers alone, came back from noise 3.

## 9. Trained long, with trimming or agreement: lessons only

`suites.py` (`trim1`, `trim_cols`, `trim_stab`; the timing suites `trim_time` to `trim_time4`), records in
`runs/trim1.jsonl`, `runs/trim_cols.jsonl` and `runs/trim_stab.jsonl`, 4 seeds per arm (2 in `trim_stab`).

**The question.** §3's grid memorizes: with blurred digits it fits the training answers through an analog code, with
snapped ones through a lookup, and one size up it is at chance. Two known remedies for memorization were tried on it.
*Trimming*: train far past the fit with weight decay. In grokking (Power et al. 2022), networks that had memorized
modular arithmetic found the general rule thousands of steps later; one account (Nanda et al. 2023) is that the
general circuit needs smaller weights than a lookup, so the decay wears the lookup away. *Agreement*: let a weight
move only where the questions agree. A memorized answer helps one question and the machine's rule helps them all, so
an update that every group of questions pushes the same way should build the rule. The AND-mask (Parascandolo et al.
2021) keeps only the gradient components on whose sign the groups agree; delta-bar-delta (Jacobs 1988; Sutton's IDBD
makes the idea incremental) gives every weight its own speed, which grows while its gradient keeps its sign from step
to step and shrinks when it flips.

**The setting.** Lessons only: GCL and, for the first time here, the step law (§2; it needs two links, so it acts in the
(3, 3, 2) runs only); nothing from the compiled rule
(no trace, no worksheet). The causal neighbourhood (§4: the setting in which the trace certified). Trained at (3, 3,
2), where every local situation occurs, in batches of 256 for 20,000 steps on a GPU; and at (3, 2, 1) on all 99
questions for 50,000 steps. §3's runs stopped after 3,000 steps, or 1,500 without progress; their best came at step
250 at (3, 3, 2) and between 900 and 1,850 at (3, 2, 1), so these budgets are 80× and 25–55× that. No early stop.
Every 500 steps a curve point records: the training share, snapped and soft; the snapped share one size up, (B, w,
k + 1) and (B, w + 1, k + 1), on a fixed sample of 500 questions, to catch a late jump; the rule's weight norm; the
sub table; and the plumbing probe below. The learning rate is 0.003 throughout. The arms:

- control: plain Adam, no weight decay;
- AdamW with decoupled weight decay 0.1, 0.3 and 1 (at this learning rate, 0.3 takes the same 0.1% off every weight
  per step as grokking's decay of 1 at a learning rate of 0.001);
- the AND-mask: the step's questions in 8 random groups; a gradient component is kept if at least 7 of the 8 groups
  agree on its sign (|mean sign| ≥ 0.75), and is then the mean gradient, zero elsewhere. The cell's table laws, which
  do not depend on the questions, join every group's gradient;
- delta-bar-delta speeds on top of Adam: a weight's speed scales its whole update; it grows by 0.02 while the weight's
  gradient has the sign of the trace of its past gradients (decay 0.9) and shrinks by 10% when it does not, within
  [0.1, 10];
- each of the last two with weight decay 0.3;
- at (3, 2, 1) only (`trim_cols`), the AND-mask across cell positions instead of questions: the grid's 4 columns are
  the groups, each column's cells run on their own copy of the rule, and a component is kept if the copies agree on
  its sign by the same threshold (|mean sign| ≥ 0.75: with 4 groups, all four, or three with the fourth at zero);
  alone and with weight decay 0.3. The cell's laws, which run on no grid, join every column's gradient.

Each arm was also started at the compiled rule instead of at random (`trim_stab`: (3, 2, 1), the rule's output
scale softened to 4, where it is still exact on the snapped and the soft path; the same lessons; 10,000 steps): an
arm that cannot keep the machine's rule once it has it cannot be expected to find it.

**The plumbing probe** (read only: it trains nothing). §3 put the grid's failure in the copies the machine needs. On
the snapped grid at the training size, the probe asks whether three of them arrive where the compiled rule reads them,
in whatever code: N's digit at a SUB cell (from the cell above), the old register at a SEL cell (from the cell above,
which in the machine holds the minuend its link started from), and the final borrow at a SEL cell (from the cell to
its right, or the cell above at the top place). For each, a lookup from the source cell's whole state to the
machine's value, fitted on half the questions and scored on the other half, as a lift over the most common value: 1
means the value can be read off the source, 0 that it is not there. On the compiled rule all three are 1.

**What happened, in plain words.** No arm certifies, and none fits the training grid on snapped digits: the best any
(3, 2, 1) run reaches at any point is 62 of its 99 questions (most peak at 40–60% of them), and the best at (3, 3, 2)
about a seventh. One size up every run stays at chance from the first curve point to the last (the best any point
reached is at most 0.07 above "always answer 0"), so there is no late jump. What the arms change is the code the grid
memorizes with:
- **Weight decay** brings the weights to a size the decay sets (a norm of about 17–20, 10–13 and 6–7 at decays 0.1,
  0.3 and 1): from step 3,000 on they stay within a fifth of where they end. At that size the cells pass blurred
  digits, and at (3, 2, 1) the blurred code fits the training answers on the soft path, though not steadily: over the
  second half of the run, decays 0.1 and 0.3 are at 0.99 or better at 41–76% of the curve points and dip as low as
  0.3–0.5 between, and decay 1 stays near 0.8 (at the last point, medians of 0.91–0.99). The snapped grid gets about
  a quarter of the answers. At decay 1 even the subtraction cell's table breaks (at (3, 2, 1), half of its 18 digit
  entries wrong in every seed); the other arms keep it whole in 58 of their 64 runs, and the worst gets 12 of the 18
  digits right.
- **The control**, without decay, fits the soft path early too (every seed by step 5,500, at a norm of 26–34), but its
  weights keep growing (to 68), its digits sharpen, and the soft fit goes: no seed is at 0.99 after step 12,000, and
  the control ends at 0.20 snapped and 0.33 soft.
- **The AND-mask** over 8 groups of questions lets a few percent of the gradient's components through (a median over
  the run of 4–8% at (3, 2, 1), 3–4% there with decay, 10–19% at (3, 3, 2)). It learns more slowly, never fits the
  soft path (0.77–0.96 at best), and ends like the control, its weights growing (to 70) and fitting neither path.
  With decay 0.3 its weights settle where decay alone puts them (a norm of 9–12), but the soft fit does not hold:
  three of four seeds reach 0.99 at some point, and they end at 0.60–0.93.
- **The AND-mask over the 4 columns** (at (3, 2, 1)) lets about as much through (6–9%). Alone, three of four seeds
  fit the soft path by step 3,000 and none holds it past step 4,000; its weights grow faster than the control's (a
  norm of 86 at the end, one seed 160), and it ends at 0.12 snapped and 0.17 soft. With decay 0.3 it ends like decay
  alone (a norm of 12–14, 0.95 soft) with fewer snapped answers (0.18).
- **Delta-bar-delta**: within the first 500 steps most weights' gradients change sign from step to step, so their
  speeds fall toward the floor (a median of 0.10–0.16 of the base rate) while a few weights speed up (the fastest at
  1–10 at the end). On most weights it is a slower Adam. Alone, its weights grow more slowly than the control's (to
  58) and it keeps the soft fit longer (to step 32,000–48,000; a median of 0.88 at the end); with decay 0.3 it ends
  like decay alone (0.27 snapped, 1.00 soft).
- **The plumbing probe** finds the three copies faintly at best: a lift of 0.0–0.3 at the end, and at most about 0.6 at
  any curve point, against 1 for the compiled rule. N's digit does not ride down to the SUB cells, the old register does
  not reach the SEL cells, and the final borrow does not come back along the row.
- **At (3, 3, 2)**, where every local situation occurs, no run fits even the soft path in 20,000 steps (0.26 at best),
  so trimming's premise, fit first and then decay, never arises there. The arms differ in their weight norms, which
  behave as at (3, 2, 1), and not in what they fit: every run ends at 0.04–0.10 snapped, about the 0.06 that "always
  answer 0" gets at that size.

At the end of the run, medians over the seeds; "its best over the run" is the best median any curve point reached:

<!-- TRIM -->
| trained at | arm | certified | training, snapped / soft | best training, snapped | one size up (k + 1 / w + 1, k + 1) | its best over the run | weight norm | sub table | plumbing: N's digit / old register / final borrow |
|---|---|---|---|---|---|---|---|---|---|
| (3, 3, 2), 20,000 steps | plain Adam (the control) | 0/4 | 0.05 / 0.06 | 0.11 | 0.06 / 0.02 | 0.07 / 0.03 | 22 → 48 | 18 + 18 | 0.03 / 0.09 / 0.16 |
| (3, 3, 2), 20,000 steps | AdamW, weight decay 0.1 | 0/4 | 0.08 / 0.22 | 0.11 | 0.05 / 0.02 | 0.07 / 0.04 | 19 → 20 | 18 + 18 | 0.09 / 0.02 / 0.16 |
| (3, 3, 2), 20,000 steps | AdamW, weight decay 0.3 | 0/4 | 0.07 / 0.24 | 0.10 | 0.05 / 0.02 | 0.07 / 0.03 | 16 → 13 | 18 + 18 | 0.06 / 0.06 / 0.15 |
| (3, 3, 2), 20,000 steps | AdamW, weight decay 1 | 0/4 | 0.06 / 0.19 | 0.08 | 0.05 / 0.02 | 0.07 / 0.03 | 10 → 7 | 11 + 16 | 0.07 / 0.03 / 0.13 |
| (3, 3, 2), 20,000 steps | AND-mask over 8 groups | 0/4 | 0.06 / 0.06 | 0.10 | 0.06 / 0.02 | 0.07 / 0.03 | 21 → 52 | 18 + 18 | 0.10 / 0.04 / 0.22 |
| (3, 3, 2), 20,000 steps | delta-bar-delta speeds | 0/4 | 0.06 / 0.14 | 0.09 | 0.05 / 0.02 | 0.07 / 0.03 | 21 → 38 | 18 + 18 | 0.04 / 0.03 / 0.20 |
| (3, 3, 2), 20,000 steps | AND-mask + weight decay 0.3 | 0/4 | 0.07 / 0.22 | 0.09 | 0.04 / 0.02 | 0.07 / 0.03 | 15 → 13 | 18 + 18 | 0.05 / 0.04 / 0.19 |
| (3, 3, 2), 20,000 steps | delta-bar-delta + weight decay 0.3 | 0/4 | 0.06 / 0.24 | 0.08 | 0.05 / 0.01 | 0.06 / 0.03 | 18 → 12 | 18 + 18 | 0.05 / 0.04 / 0.16 |
| (3, 2, 1), 50,000 steps | plain Adam (the control) | 0/4 | 0.20 / 0.33 | 0.56 | 0.12 / 0.05 | 0.21 / 0.08 | 21 → 68 | 18 + 18 | 0.29 / 0.04 / 0.28 |
| (3, 2, 1), 50,000 steps | AdamW, weight decay 0.1 | 0/4 | 0.26 / 0.96 | 0.51 | 0.13 / 0.04 | 0.22 / 0.08 | 19 → 17 | 18 + 18 | 0.14 / 0.09 / 0.28 |
| (3, 2, 1), 50,000 steps | AdamW, weight decay 0.3 | 0/4 | 0.29 / 0.99 | 0.44 | 0.14 / 0.05 | 0.19 / 0.07 | 15 → 11 | 18 + 18 | 0.11 / 0.07 / 0.11 |
| (3, 2, 1), 50,000 steps | AdamW, weight decay 1 | 0/4 | 0.25 / 0.91 | 0.38 | 0.11 / 0.04 | 0.20 / 0.08 | 9 → 7 | 9 + 16 | 0.11 / 0.01 / 0.17 |
| (3, 2, 1), 50,000 steps | AND-mask over 8 groups | 0/4 | 0.19 / 0.24 | 0.53 | 0.15 / 0.05 | 0.21 / 0.08 | 21 → 70 | 18 + 18 | 0.32 / 0.06 / 0.20 |
| (3, 2, 1), 50,000 steps | delta-bar-delta speeds | 0/4 | 0.28 / 0.88 | 0.53 | 0.14 / 0.04 | 0.21 / 0.08 | 21 → 58 | 18 + 18 | 0.29 / 0.10 / 0.25 |
| (3, 2, 1), 50,000 steps | AND-mask + weight decay 0.3 | 0/4 | 0.34 / 0.76 | 0.45 | 0.15 / 0.05 | 0.21 / 0.08 | 15 → 12 | 17 + 18 | 0.10 / 0.04 / 0.23 |
| (3, 2, 1), 50,000 steps | delta-bar-delta + weight decay 0.3 | 0/4 | 0.27 / 1.00 | 0.37 | 0.13 / 0.05 | 0.20 / 0.07 | 20 → 12 | 18 + 18 | 0.12 / -0.01 / 0.25 |
| (3, 2, 1), 50,000 steps | AND-mask over the 4 columns | 0/4 | 0.12 / 0.17 | 0.50 | 0.17 / 0.05 | 0.21 / 0.08 | 22 → 86 | 18 + 18 | 0.15 / 0.01 / 0.17 |
| (3, 2, 1), 50,000 steps | AND-mask over the columns + weight decay 0.3 | 0/4 | 0.18 / 0.95 | 0.40 | 0.14 / 0.04 | 0.21 / 0.08 | 16 → 13 | 18 + 18 | 0.11 / 0.08 / 0.22 |
<!-- /TRIM -->

Over training: the training share snapped / soft / one size up ((B, w + 1, k + 1), on the curve's 500-question
sample), medians over the seeds:

<!-- TRIM_CURVE -->
| trained at (3, 3, 2); one size up is (3, 4, 3) | step 1,000 | step 5,000 | step 10,000 | step 20,000 |
|---|---|---|---|---|
| plain Adam (the control) | 0.07 / 0.18 / 0.02 | 0.08 / 0.18 / 0.01 | 0.05 / 0.08 / 0.02 | 0.05 / 0.06 / 0.03 |
| AdamW, weight decay 0.1 | 0.08 / 0.18 / 0.02 | 0.07 / 0.19 / 0.02 | 0.08 / 0.23 / 0.02 | 0.08 / 0.22 / 0.03 |
| AdamW, weight decay 0.3 | 0.08 / 0.19 / 0.02 | 0.07 / 0.21 / 0.02 | 0.08 / 0.22 / 0.02 | 0.07 / 0.24 / 0.02 |
| AdamW, weight decay 1 | 0.06 / 0.18 / 0.02 | 0.05 / 0.19 / 0.01 | 0.06 / 0.19 / 0.02 | 0.06 / 0.19 / 0.02 |
| AND-mask over 8 groups | 0.07 / 0.18 / 0.01 | 0.07 / 0.12 / 0.01 | 0.07 / 0.17 / 0.01 | 0.06 / 0.06 / 0.02 |
| delta-bar-delta speeds | 0.07 / 0.18 / 0.02 | 0.07 / 0.20 / 0.02 | 0.06 / 0.19 / 0.01 | 0.06 / 0.14 / 0.02 |
| AND-mask + weight decay 0.3 | 0.06 / 0.18 / 0.02 | 0.07 / 0.22 / 0.01 | 0.06 / 0.23 / 0.02 | 0.07 / 0.22 / 0.02 |
| delta-bar-delta + weight decay 0.3 | 0.07 / 0.18 / 0.02 | 0.06 / 0.21 / 0.02 | 0.06 / 0.21 / 0.02 | 0.06 / 0.24 / 0.02 |

| trained at (3, 2, 1); one size up is (3, 3, 2) | step 1,000 | step 10,000 | step 25,000 | step 50,000 |
|---|---|---|---|---|
| plain Adam (the control) | 0.37 / 0.73 / 0.05 | 0.36 / 0.83 / 0.04 | 0.26 / 0.56 / 0.04 | 0.20 / 0.33 / 0.06 |
| AdamW, weight decay 0.1 | 0.38 / 0.91 / 0.05 | 0.31 / 0.98 / 0.04 | 0.36 / 0.98 / 0.04 | 0.26 / 0.96 / 0.03 |
| AdamW, weight decay 0.3 | 0.29 / 0.72 / 0.05 | 0.34 / 0.99 / 0.03 | 0.25 / 1.00 / 0.03 | 0.29 / 0.99 / 0.04 |
| AdamW, weight decay 1 | 0.24 / 0.56 / 0.07 | 0.28 / 0.71 / 0.05 | 0.14 / 0.91 / 0.04 | 0.25 / 0.91 / 0.04 |
| AND-mask over 8 groups | 0.29 / 0.47 / 0.05 | 0.33 / 0.71 / 0.06 | 0.38 / 0.50 / 0.06 | 0.19 / 0.24 / 0.04 |
| delta-bar-delta speeds | 0.27 / 0.96 / 0.06 | 0.33 / 0.99 / 0.05 | 0.31 / 0.96 / 0.05 | 0.28 / 0.88 / 0.04 |
| AND-mask + weight decay 0.3 | 0.30 / 0.58 / 0.06 | 0.25 / 0.70 / 0.05 | 0.27 / 0.75 / 0.04 | 0.34 / 0.76 / 0.04 |
| delta-bar-delta + weight decay 0.3 | 0.29 / 0.90 / 0.07 | 0.21 / 1.00 / 0.05 | 0.22 / 1.00 / 0.06 | 0.27 / 1.00 / 0.04 |
| AND-mask over the 4 columns | 0.35 / 0.86 / 0.05 | 0.30 / 0.45 / 0.05 | 0.18 / 0.18 / 0.05 | 0.12 / 0.17 / 0.05 |
| AND-mask over the columns + weight decay 0.3 | 0.30 / 0.66 / 0.06 | 0.27 / 0.74 / 0.05 | 0.20 / 0.88 / 0.05 | 0.18 / 0.95 / 0.04 |
<!-- /TRIM_CURVE -->

**Why trimming does not work here.** In grokking's account the general rule is the cheaper one in weight: decay wears
down a lookup that needs larger weights than the rule, until the rule is what is left. Here it is the other way round.
The machine's digits must stay sharp through every tick (17 at (3, 2, 1), 50 at (3, 3, 2)), and on the soft path, the
one the lessons train through, that takes large weights. The compiled rule written into the trim runs' network
(`sharpness.py`) is exact on both paths at (3, 2, 1) only from an output scale of 3, a weight norm of 37 (from 4, a
norm of 45, at (3, 3, 2)); at smaller scales it is still exact when snapped, but its states blur and the soft path gets
0.18 of training right. The decayed runs at (3, 2, 1) end at norms of 6–19, and their cells pass blurred states. Per
channel, the
median over every tick and written cell of a state's largest probability, and the share of states above 0.9, at
(3, 2, 1) at the end of the run:

<!-- SHARP -->
| rule | weight norm | training, snapped / soft | d | n | b | m | g |
|---|---|---|---|---|---|---|---|
| the compiled rule, output scale 4 | 45 | 1.00 / 1.00 | 0.95 (100%) | 0.95 (100%) | 0.98 (100%) | 0.96 (100%) | 0.98 (100%) |
| the compiled rule, output scale 20 | 198 | 1.00 / 1.00 | 1.00 (100%) | 1.00 (100%) | 1.00 (100%) | 1.00 (100%) | 1.00 (100%) |
| plain Adam (the control) (4 seeds) | 68 | 0.20 / 0.33 | 1.00 (89%) | 1.00 (96%) | 1.00 (97%) | 1.00 (93%) | 1.00 (96%) |
| AdamW, weight decay 0.1 (4 seeds) | 17 | 0.26 / 0.96 | 0.92 (55%) | 0.60 (7%) | 0.87 (43%) | 0.69 (12%) | 0.80 (32%) |
| AdamW, weight decay 0.3 (4 seeds) | 12 | 0.29 / 0.99 | 0.84 (39%) | 0.54 (0%) | 0.84 (39%) | 0.65 (4%) | 0.71 (9%) |
| AdamW, weight decay 1 (4 seeds) | 6 | 0.25 / 0.91 | 0.62 (16%) | 0.54 (4%) | 0.82 (33%) | 0.63 (8%) | 0.64 (13%) |
| AND-mask over 8 groups (4 seeds) | 70 | 0.19 / 0.24 | 1.00 (88%) | 1.00 (100%) | 1.00 (99%) | 1.00 (96%) | 1.00 (99%) |
| delta-bar-delta speeds (4 seeds) | 58 | 0.28 / 0.88 | 1.00 (91%) | 0.99 (78%) | 1.00 (93%) | 0.99 (76%) | 1.00 (88%) |
| AND-mask + weight decay 0.3 (4 seeds) | 12 | 0.34 / 0.76 | 0.77 (27%) | 0.54 (2%) | 0.89 (48%) | 0.68 (4%) | 0.74 (9%) |
| delta-bar-delta + weight decay 0.3 (4 seeds) | 12 | 0.27 / 1.00 | 0.80 (34%) | 0.49 (0%) | 0.91 (52%) | 0.60 (2%) | 0.66 (0%) |
| AND-mask over the 4 columns (4 seeds) | 86 | 0.12 / 0.17 | 1.00 (91%) | 1.00 (100%) | 1.00 (99%) | 1.00 (100%) | 1.00 (100%) |
| AND-mask over the columns + weight decay 0.3 (4 seeds) | 13 | 0.18 / 0.95 | 0.80 (32%) | 0.49 (0%) | 0.95 (65%) | 0.56 (0%) | 0.68 (5%) |
<!-- /SHARP -->

With decay, N's digit and the memo pass as blurs (a median largest probability of 0.5–0.7; at most 12% of states above
0.9), and that is the code that fits the training answers. Without decay the weights grow (to 58–86 in norm), the
states sharpen nearly to the compiled rule's, and the soft fit goes: the control and both AND-masks end fitting
neither path, and delta-bar-delta, whose weights grow the slowest and whose states stay a little blurrier, keeps most
of it.
The control shows the same over time: it fits the soft path while its norm is 26–39 and loses the fit as the norm
grows. So decay does not wear the lookup away: it makes the blurred code the cheap solution and the machine the
expensive one.

**Started at the machine's rule, no arm keeps it.** `trim_stab` started every arm at the compiled rule (output scale
4, a norm of 45, exact on both paths) with the same lessons, for 10,000 steps:

<!-- TRIM_STAB -->
| arm | first step below exact, snapped (each seed) | the same, soft | training at the end, snapped / soft | weight norm | one size up at the end (k + 1 / w + 1, k + 1) | certified | sub table |
|---|---|---|---|---|---|---|---|
| plain Adam (the control) | 200 and 200 | 500 and 1,000 | 0.23 / 0.42 | 47 → 54 | 0.12 / 0.05 | 0/2 | 18 + 18 |
| AdamW, weight decay 0.1 | 100 and 100 | 1,500 and 500 | 0.26 / 0.84 | 41 → 20 | 0.10 / 0.04 | 0/2 | 18 + 18 |
| AdamW, weight decay 0.3 | 200 and 100 | 1,500 and 500 | 0.19 / 0.88 | 32 → 14 | 0.16 / 0.05 | 0/2 | 18 + 18 |
| AdamW, weight decay 1 | 100 and 200 | 500 and 500 | 0.23 / 0.67 | 15 → 6 | 0.15 / 0.04 | 0/2 | 7 + 16 |
| AND-mask over 8 groups | 100 and 100 | 1,000 and 7,500 | 0.21 / 0.24 | 47 → 53 | 0.18 / 0.06 | 0/2 | 18 + 18 |
| delta-bar-delta speeds | 300 and 100 | 2,500 and 6,000 | 0.21 / 0.54 | 49 → 63 | 0.12 / 0.04 | 0/2 | 18 + 18 |
| AND-mask + weight decay 0.3 | 100 and 100 | 2,000 and 1,500 | 0.13 / 0.46 | 32 → 12 | 0.07 / 0.03 | 0/2 | 15 + 16 |
| delta-bar-delta + weight decay 0.3 | 200 and 200 | 500 and 3,000 | 0.27 / 1.00 | 34 → 15 | 0.14 / 0.05 | 0/2 | 18 + 18 |
| AND-mask over the 4 columns | 100 and 100 | 500 and 4,000 | 0.17 / 0.31 | 48 → 57 | 0.17 / 0.06 | 0/2 | 18 + 18 |
| AND-mask over the columns + weight decay 0.3 | 100 and 100 | 1,500 and 1,000 | 0.20 / 0.81 | 32 → 14 | 0.12 / 0.04 | 0/2 | 18 + 18 |
<!-- /TRIM_STAB -->

Every arm, the control included, is below exact on the snapped path within 100–300 steps, and none comes back (0.13–0.27
at the end). The soft fit lasts longer in some runs (to step 7,500 in one AND-mask run), so for a while the grid is
exact on blurred digits and not on snapped ones: the lessons, which only see the soft path, are met as well by blurred
neighbours of the rule. Then the runs end where the random starts do: with decay the weights shrink to the size decay
sets (a norm of 6–20) and the grid keeps a soft fit on blurred digits (0.46–1.00 of training), and without decay the
weights grow and the soft fit goes too (0.24–0.54). Two causes push the rule off, and `rest.py` separates them:
- at output scale 4 the lessons are met but not perfectly (each lesson's loss is 0.04–0.08), and their gradient leads
  off the rule as well as along it;
- at output scale 20 every lesson's loss is 0 and its gradient about 1e-8, and plain Adam still leaves the rule: 0.86
  of training right at step 75, 0.61 at step 100. Adam divides each weight's step by the size of that weight's recent
  gradients, so a gradient of 1e-8 moves every weight by about half the learning rate, and the compiled rule's
  thresholds sit exactly at the edge (one unmet condition puts a unit at 0, the edge of its ReLU). With Adam's eps
  raised from 1e-8 to 1e-4 the same run stays exact.

A rule the training cannot hold is not one it can be expected to find. The arms tested here all sit on top of Adam at
a learning rate of 0.003, and none of them changes that; §10 tries two optimizers that can rest at the rule.

**What was tried, and how hard** (dead ends):
- Weight decay at 0.1, 0.3 and 1, 4 seeds each at both sizes: 50,000 steps at (3, 2, 1) is 9–33 times the steps the
  soft fit took (in the runs that reached 0.99), and from step 3,000 on the weight norm stayed within a fifth of where
  it ended.
- The AND-mask over 8 random groups of questions (threshold 7 of 8), alone and with decay 0.3, 4 seeds each at both
  sizes; over the grid's 4 columns, alone and with decay 0.3, 4 seeds each at (3, 2, 1).
- Delta-bar-delta speeds, alone and with decay 0.3, 4 seeds each at both sizes.
- Every arm from the compiled rule, 2 seeds each, 10,000 steps.
- In all, 72 runs from random (40 at (3, 2, 1), 32 at (3, 3, 2)) and 20 from the compiled rule.
- Not tried here: another optimizer than Adam (§10 tries plain SGD and Adam with a larger eps, which can rest at the
  rule), another learning rate, more seeds (no arm certified, so none got the 8 the plan held for one that did),
  agreement across seeds or sizes rather than question groups or columns, and training on the snapped path (STE) with
  decay.

**Budget notes.**
- At (3, 3, 2) no arm fit training even on the soft path in 20,000 steps (the best curve point: 0.26 soft): trimming's
  premise, fit first and then decay, never happened at that size. How long the fit would take is open.
- (3, 3, 2) does not fit in 8 GB on CPU with the step law (it ran out of memory in the timing suite), so it ran on L4
  GPUs, three runs to a GPU, each in its own process: 0.8–0.9 s a step, or 1.1 s beside an AND-mask run, which itself
  took 1.4 s (4.5–6.1 hours a run, 7.6–8.0 for an AND-mask run). The AND-mask is 8 forward-backward passes per step;
  done as one grouped convolution over the 8 copies of the rule it costs 1.0 s a step on an L4 alone (6.2 s group by
  group).
- (3, 2, 1) ran on 4-core CPU containers: 0.23–0.28 s a step plain, 0.4 s with the AND-mask, about 0.6 s over the
  columns. In the sweep a run took 3.1–4.3 hours (8.1 for one resumed on a container four times slower), 4.6–6.6 with
  the AND-mask, 7.1–8.5 over the columns.
- Modal preempted CPU containers often (up to four times in an eight-hour run). Every run saved itself every 500
  steps and resumed exactly where it stopped (a resumed run equals an uninterrupted one). A start guard meant to stop
  crash loops counted those restarts and ended 10 runs; they were relaunched and resumed from their checkpoints, and
  the guard now counts only starts that made no progress.
- One more run, decay 0.1 seed 1 at (3, 2, 1), was then on its second start. It saved its checkpoint at step 47,500
  and wrote nothing after, most likely ended by the same guard on a third start; its curve to there is kept in
  `records/trim_cut.json` (from `cut.py`). Once the other runs had ended it was resumed from that checkpoint in a
  launch of its own and finished: its record's curve up to step 47,500 equals the kept one.

## 10. An optimizer that can rest at the rule: lessons only

`suites.py` (`rest_lr`, `rest1`, `rest_stab`, `rest_stab3`), records in `runs/rest_lr.jsonl`, `runs/rest1.jsonl`,
`runs/rest_stab.jsonl` and `runs/rest_stab3.jsonl`, 4 seeds per arm (one per learning rate in `rest_lr`); `rest.py`,
`erosion.py`, `cut.py`.

**The question.** Every arm of §9 sat on Adam, and Adam does not stay at an exact rule: it divides each weight's step
by the size of that weight's recent gradients, so even where every lesson is met, with gradients of 1e-8, it moves
every weight by about half the learning rate (`rest.py`). Two optimizers do stay there. Adam with eps raised to 1e-4
shrinks the step of any weight whose gradient is well under 1e-4, and plain SGD's step is the gradient times the
learning rate. With them, two questions: started at the compiled rule, as in `trim_stab`, does the grid keep the rule;
and from random, does it find the rule, or memorize as it did with Adam?

**The setting.** §9's at (3, 2, 1): lessons only (GCL; the step law needs two links and does not act at k = 1), the
causal neighbourhood, all 99 questions every step, a curve point every 500 steps, no early stop. The arms:
- Adam with eps 1e-4 at §9's learning rate (0.003), alone and as AdamW with weight decay 0.3;
- plain SGD at learning rate 0.01, alone and with weight decay (L2, set to take the same 0.09% off every weight per
  step as AdamW 0.3 does at 0.003);
- plain SGD at learning rate 0.1, the same two ways.

SGD's learning rates come from `rest_lr` (one seed each, below): 0.01 is the largest of those tried that kept the
compiled rule's training answers for 10,000 steps, and 0.1 learned fastest from random. Each arm ran from random for
50,000 steps (`rest1`), and from the compiled rule for 10,000 steps (`rest_stab`) at two output scales: 4, as in
`trim_stab`, where the lessons are met but not perfectly (a tenth of the weights have a gradient of 1e-3 or more,
`rest.py grads`); and 20, where every lesson is met and no gradient is above 1.1e-7 (with plain Adam and AdamW 0.3
there too, which `trim_stab` ran only at scale 4). Because of what the start at scale 4 showed, it was repeated at
(3, 3, 2), where almost every local situation of a bigger grid occurs (`rest_stab3`: §9's (3, 3, 2) lessons, the step
law included, in batches of 256 on GPU; output scale 4, the smallest at which the compiled rule is exact there on both
paths; 10,000 steps): plain Adam, Adam with eps 1e-4, and plain SGD at 0.003 and 0.01.

13 of `rest1`'s 24 runs were cut off: at about 07:25 PT on 2026-10-24 Modal stopped every container of the workspace
when it reached its spend limit, with these runs at steps 18,000–46,500 of their 50,000 (each run's step is in the
table of cut-off runs below). Their curves up to there are kept (`records/rest_cut.json`, from `cut.py`), and their
checkpoints are on the Volume; the tables of the runs' ends have the 11 that finished. Every other run here finished.

**SGD's learning rate** (`rest_lr`): from random, the training share snapped / soft; from the compiled rule at output
scale 4, when it first fell below exact:

<!-- REST_LR -->
| learning rate | from random: training at step 1,000, snapped / soft | at step 5,000 | weight norm at step 5,000 | from the compiled rule: first step below exact, snapped / soft | training at step 10,000 |
|---|---|---|---|---|---|
| 0.003 | · | · | · | never / never | 1.00 / 1.00 |
| 0.01 | 0.16 / 0.18 | 0.31 / 0.95 | 21 | never / never | 1.00 / 1.00 |
| 0.02 | · | · | · | 7,600 / 8,000 | 0.00 / 0.00 |
| 0.03 | 0.16 / 0.39 | 0.27 / 0.32 | 29 | · | · |
| 0.1 | 0.38 / 0.51 | 0.60 / 0.92 | 37 | · | · |
| 0.3 | 0.19 / 0.18 | 0.19 / 0.18 | 27 | · | · |
| 1 | 0.12 / 0.12 | 0.12 / 0.12 | 165 | · | · |
<!-- /REST_LR -->

**What happened, in plain words.**
- **Where every lesson is met (output scale 20), both optimizers rest at the rule, and nothing else does.** Adam with
  eps 1e-4 and plain SGD, at 0.01 and at 0.1, keep it exactly for the 10,000 steps and certify, 4 of 4 seeds each:
  with gradients of at most 1.1e-7 nothing moves them. Plain Adam is below exact within 100 steps, as `rest.py`
  showed. Every arm with weight decay leaves too (AdamW 0.3 by step 100–300, as plain Adam; with eps 1e-4 at step 500;
  SGD at 0.1 and 0.01 at steps 700 and 1,100), though decay alone does not break the rule: shrinking every weight of
  the compiled rule by one factor keeps its snapped path exact at any factor (it has no output bias, so its outputs
  only scale) and its soft path down to a norm of 79, from 198 (`rest.py shrink`). What the shrinking does is bring
  the lessons' gradient back: its largest is 1.1e-7 at the start, 8e-3 once every weight is at 0.64 of its size
  (where 0.09% a step puts it at step 500) and 0.17 at half, against 0.21 at output scale 4. Decay walks the rule out
  of the one place where it can rest.
- **Where the lessons still pull (output scale 4), only plain SGD at a small learning rate keeps the training
  answers.** At 0.01 it keeps all 99 on both paths for the 10,000 steps, in every seed, and at 0.003 too (`rest_lr`;
  at 0.02 it holds until step 7,600, then answers every question wrong). The seeds differ only in the extra ticks drawn
  each step, and take nearly one path: each moves its weights 2.97 from the start, and they end within 0.004–0.013 of
  one another. Adam with eps 1e-4 is below exact by step 100–200, as plain Adam was: at output scale 4 the gradients
  that matter are of 1e-3 and more, and eps 1e-4 does not shrink their steps. Every arm with weight decay is below
  exact by step 100–300, its weights shrinking (to a norm of 7–12 by the end). SGD at 0.1 answers every question wrong
  by step 100.
- **It keeps the answers, not the rule.** One size up, SGD at 0.01 is exact until step 2,500 and then slips: 0.99 /
  0.99 at step 3,000, 0.98 / 0.97 at step 5,000, 0.70 / 0.64 at step 10,000, while every training answer stays right.
  A smaller learning rate only slows this. SGD at 0.003 follows 0.01's path, 0.3 times as fast: at its step 10,000 it
  has the lessons' loss and the weight norm that 0.01 had at step 3,000 (0.038 against 0.037; 45.88 at both), and it
  is still whole one size up there, about where 0.01 began to slip.
- **Where it slips is where the training grid never goes** (`erosion.py`, on seed 0 at step 10,000). Run for one tick
  from each of the machine's own states, the drifted rule acts as the machine wherever the situation (the kinds of the
  3 × 3 neighbourhood and the five states the causal rule reads) occurs at (3, 2, 1): it is wrong on 0.0043% of those
  cell-ticks at (3, 2, 2) and 0.0025% at (3, 3, 2) (0.016% at the training size), and on 3.3% and 2.4% of the cell-ticks
  whose situation (3, 2, 1) never shows. Those are 31% of the cell-ticks at (3, 2, 2) and 47% at (3, 3, 2), and they
  hold 99.7–99.9% of its errors. The lessons pin the rule only where the training grid goes, and their gradient, which
  still has something to improve on the soft path, moves the rest freely. This is §4's "situations never met" again,
  now pulling the machine's own rule away from itself.
- **At (3, 3, 2) nothing keeps the rule for 10,000 steps.** Plain Adam and Adam with eps 1e-4 are below exact within
  100–400 steps. Plain SGD at 0.003 and at 0.01 holds it exactly, on training and one size up, while the lessons' loss
  keeps falling (at 0.003 from 0.26–0.27 at step 100 to 0.07–0.09, and to 0.016 in the longest run at 0.01). Then,
  within 100 steps, the loss jumps 47- to 3,000-fold, and the grid falls to 0.00–0.08 of its training answers (about
  what "always answer 0" gets, 0.06) and never comes back: at step 2,300–4,300 at 0.003, and 700–8,900 at 0.01. At
  0.003 the weights hardly move in the fall (the norm changes by 0.03–0.19 across it in three of the four seeds). One
  run lasted long enough to show (3, 2, 1)'s drift: at 0.01, seed 3 began missing training answers at step 4,500 (2
  of the 2,835; 50 by step 8,500), and one size up from step 5,000 (0.98 / 0.93 at step 8,500), before it fell at
  step 8,900. Why SGD falls here at rates that hold at (3, 2, 1) is not settled: the batches of 256 (at (3, 2, 1)
  every step sees all 99 questions), the step law and the GPU (whose convolutions may run in TF32) all differ. Whether
  a training grid that shows almost every situation stops the drift is therefore open too.
- **From random nothing changes: every arm memorizes.** One size up no curve point of any of the 24 runs (the cut-off
  ones up to where they stopped) is more than 0.06 above "always answer 0" (at best 0.24 at (3, 2, 2) and 0.10 at (3,
  3, 2), against 0.18 and 0.06), and none fits training on snapped digits: the best any run reaches at any curve point
  is 63 of its 99 questions. The arms differ, as in §9, in the code they memorize with:
  - Adam with eps 1e-4 behaves as §9's control: every seed fits the soft path by step 1,000–3,500 (at a norm of
    24–31), its weights keep growing (to 60–67), and the fit goes (never at 0.99 in the second half of the run;
    0.18–0.76 at the last point);
  - plain SGD at 0.01 fits the soft path by step 4,500–7,000 and keeps it, at 0.99 or better at 75–100% of the curve
    points in the second half (§9's steadiest arms: 66–82%), at a norm of 26–27 and a third of the answers snapped
    (0.29–0.40). Its states are sharper than decay leaves them and well short of the rule's (40–86% of them above 0.9
    in the seed that finished, against all of them for the compiled rule): the steadiest blurred code yet;
  - plain SGD at 0.1 fits the soft path in two seeds (by step 3,000 and 6,000), then every seed collapses to a
    constant answer (0.12 or 0.18 of training on both paths, from step 9,000–27,500), its weight norm jumping to 47–93;
  - with weight decay the weights settle where decay puts them (a norm of 7 with SGD at 0.01, 11–13 otherwise) and the
    code is §9's blurred one: AdamW keeps the soft fit at 56–79% of the second half's curve points, SGD at 0.1 reaches
    0.99 at only 1 and 4 curve points in two of its seeds, and SGD at 0.01 never (0.66–0.76 at best).

  The plumbing probe finds the three copies as faintly as in §9: at the end, medians of 0.00–0.29 and at most 0.42 in
  any run that finished, against 1 for the compiled rule.

**What finishing the 13 cut-off runs can change.** Nothing started at the compiled rule: all 75 of those runs, at
(3, 2, 1) and at (3, 3, 2), finished, as did `rest_lr`'s 5 from random. Nor the from-random verdict, that no arm finds
the rule: every arm has at least one seed that ran its 50,000 steps at chance one size up throughout, and the cut-off
seeds had shown nothing different by where they stopped; finishing them could add only a late jump in their last
3,500–32,000 steps, which none of the 11 finished runs made. What can change are the from-random numbers read at the
end of a run: the medians in the table of the runs' ends and in the curve table's last column (now over 1–3 seeds per
arm), the sharpness of plain SGD at 0.01 (one seed so far), and how steadily each arm holds its soft fit, which for a
cut-off run is counted over the second half of the curve it reached.

Started at the compiled rule at (3, 2, 1) (`rest_stab`; as `trim_stab`'s table):

<!-- REST_STAB -->
| arm | first step below exact, snapped (each seed) | the same, soft | training at the end, snapped / soft | weight norm | one size up at the end (k + 1 / w + 1, k + 1) | certified | sub table |
|---|---|---|---|---|---|---|---|
| scale 4: Adam, eps 1e-4 | 200, 100, 200 and 100 | 5,500, 5,500, 2,000 and 500 | 0.33 / 0.61 | 47 → 53 | 0.17 / 0.05 | 0/4 | 18 + 18 |
| scale 4: AdamW, eps 1e-4, weight decay 0.3 | 300, 100, 100 and 200 | 1,000, 1,000, 500 and 500 | 0.21 / 0.93 | 32 → 12 | 0.14 / 0.05 | 0/4 | 17 + 17 |
| scale 4: plain SGD, learning rate 0.01 | never, never, never and never | never, never, never and never | 1.00 / 1.00 | 46 → 46 | 0.70 / 0.64 | 0/4 | 18 + 18 |
| scale 4: plain SGD 0.01 + weight decay | 300, 300, 300 and 300 | 500, 500, 500 and 500 | 0.24 / 0.59 | 29 → 7 | 0.14 / 0.05 | 0/4 | 11 + 16 |
| scale 4: plain SGD, learning rate 0.1 | 100, 100, 100 and 100 | 500, 500, 500 and 500 | 0.00 / 0.00 | 52 → 52 | 0.00 / 0.00 | 0/4 | 6 + 9 |
| scale 4: plain SGD 0.1 + weight decay | 100, 100, 100 and 100 | 500, 500, 500 and 500 | 0.19 / 0.36 | 34 → 11 | 0.12 / 0.05 | 0/4 | 5 + 9 |
| scale 20: plain Adam (the control) | 100, 100, 100 and 100 | 500, 500, 500 and 500 | 0.18 / 0.18 | 198 → 199 | 0.18 / 0.06 | 0/4 | 16 + 18 |
| scale 20: AdamW, weight decay 0.3 | 100, 100, 200 and 300 | 500, 500, 500 and 500 | 0.18 / 0.97 | 127 → 12 | 0.14 / 0.04 | 0/4 | 17 + 18 |
| scale 20: Adam, eps 1e-4 | never, never, never and never | never, never, never and never | 1.00 / 1.00 | 198 → 198 | 1.00 / 1.00 | 4/4 | 18 + 18 |
| scale 20: AdamW, eps 1e-4, weight decay 0.3 | 500, 500, 500 and 500 | 500, 500, 500 and 500 | 0.23 / 0.88 | 126 → 11 | 0.13 / 0.05 | 0/4 | 15 + 17 |
| scale 20: plain SGD, learning rate 0.01 | never, never, never and never | never, never, never and never | 1.00 / 1.00 | 198 → 198 | 1.00 / 1.00 | 4/4 | 18 + 18 |
| scale 20: plain SGD 0.01 + weight decay | 1,100, 1,100, 1,100 and 1,100 | 1,500, 1,500, 1,500 and 1,500 | 0.21 / 0.38 | 126 → 6 | 0.12 / 0.05 | 0/4 | 11 + 16 |
| scale 20: plain SGD, learning rate 0.1 | never, never, never and never | never, never, never and never | 1.00 / 1.00 | 198 → 198 | 1.00 / 1.00 | 4/4 | 18 + 18 |
| scale 20: plain SGD 0.1 + weight decay | 700, 700, 700 and 700 | 1,000, 1,000, 1,000 and 1,000 | 0.18 / 0.19 | 126 → 5 | 0.16 / 0.06 | 0/4 | 5 + 9 |
<!-- /REST_STAB -->

One size up, for the starts that ended with every training answer right (k + 1 / w + 1, k + 1, on the curve's sample,
medians over the seeds; learning rate 0.003 from `rest_lr`):

<!-- REST_UP -->
| started at the compiled rule | seeds | step 2,500 | step 5,000 | step 7,500 | step 10,000 |
|---|---|---|---|---|---|
| (3, 2, 1), output scale 4: plain SGD, learning rate 0.003 | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| (3, 2, 1), output scale 4: plain SGD, learning rate 0.01 | 4 | 1.00 / 1.00 | 0.98 / 0.97 | 0.97 / 0.93 | 0.70 / 0.64 |
| (3, 2, 1), output scale 20: Adam, eps 1e-4 | 4 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| (3, 2, 1), output scale 20: plain SGD, learning rate 0.01 | 4 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| (3, 2, 1), output scale 20: plain SGD, learning rate 0.1 | 4 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
<!-- /REST_UP -->

Where the drifted rule goes wrong (`erosion.py`: plain SGD at 0.01, seed 0, at step 10,000; one tick of it from each
state of the machine's own run from blank, over 1.5 times the depth's ticks; the rates are per cell-tick):

<!-- EROSION -->
| rule | measured at | answers right | cell-ticks in situations never met at the training size | one-tick errors in situations met there (rate) | in situations never met (rate) |
|---|---|---|---|---|---|
| plain SGD, learning rate 0.01, 10,000 steps from the compiled rule (output scale 4) | (3, 2, 1), the training size, 99 questions | 1.00 | · | 8 (0.016%) | · |
| | (3, 2, 2), every one of 297 questions | 0.70 | 31% | 28 (0.0043%) | 9,605 (3.3%) |
| | (3, 3, 2), a sample of 200 | 0.61 | 47% | 12 (0.0025%) | 9,965 (2.4%) |
<!-- /EROSION -->

At (3, 3, 2) (`rest_stab3`):

<!-- REST_STAB3 -->
| arm | first step below exact, snapped (each seed) | the same, soft | training at the end, snapped / soft | weight norm | one size up at the end (k + 1 / w + 1, k + 1) | certified | sub table |
|---|---|---|---|---|---|---|---|
| plain Adam (the control) | 100, 100, 200 and 100 | 500, 500, 500 and 500 | 0.06 / 0.06 | 47 → 52 | 0.06 / 0.02 | 0/4 | 18 + 18 |
| Adam, eps 1e-4 | 100, 100, 400 and 100 | 500, 500, 500 and 500 | 0.06 / 0.06 | 46 → 52 | 0.06 / 0.03 | 0/4 | 17 + 18 |
| plain SGD, learning rate 0.003 | 4,300, 2,800, 2,300 and 2,600 | 4,500, 3,000, 2,500 and 3,000 | 0.06 / 0.11 | 46 → 46 | 0.06 / 0.02 | 0/4 | 17 + 18 |
| plain SGD, learning rate 0.01 | 2,800, 700, 1,700 and 4,500 | 3,000, 1,000, 2,000 and 9,000 | 0.06 / 0.06 | 46 → 55 | 0.06 / 0.03 | 0/4 | 6 + 9 |
<!-- /REST_STAB3 -->

From random, at the end of the run, medians over the runs that finished ("certified" counts them):

<!-- REST -->
| trained at | arm | certified | training, snapped / soft | best training, snapped | one size up (k + 1 / w + 1, k + 1) | its best over the run | weight norm | sub table | plumbing: N's digit / old register / final borrow |
|---|---|---|---|---|---|---|---|---|---|
| (3, 2, 1), 50,000 steps | Adam, eps 1e-4 | 0/2 | 0.20 / 0.29 | 0.54 | 0.13 / 0.06 | 0.22 / 0.08 | 21 → 63 | 18 + 18 | 0.29 / 0.10 / 0.22 |
| (3, 2, 1), 50,000 steps | AdamW, eps 1e-4, weight decay 0.3 | 0/2 | 0.33 / 0.89 | 0.52 | 0.15 / 0.05 | 0.21 / 0.07 | 16 → 12 | 18 + 18 | 0.12 / 0.02 / 0.20 |
| (3, 2, 1), 50,000 steps | plain SGD, learning rate 0.01 | 0/1 | 0.40 / 1.00 | 0.51 | 0.21 / 0.06 | 0.21 / 0.08 | 17 → 27 | 18 + 18 | 0.09 / 0.14 / 0.26 |
| (3, 2, 1), 50,000 steps | plain SGD 0.01 + weight decay | 0/3 | 0.17 / 0.52 | 0.38 | 0.10 / 0.03 | 0.23 / 0.08 | 11 → 7 | 15 + 17 | 0.12 / 0.04 / 0.16 |
| (3, 2, 1), 50,000 steps | plain SGD, learning rate 0.1 | 0/2 | 0.15 / 0.15 | 0.59 | 0.15 / 0.05 | 0.21 / 0.07 | 18 → 82 | 6 + 9 | 0.00 / 0.00 / 0.00 |
| (3, 2, 1), 50,000 steps | plain SGD 0.1 + weight decay | 0/1 | 0.42 / 0.80 | 0.59 | 0.19 / 0.06 | 0.21 / 0.07 | 12 → 12 | 8 + 11 | 0.07 / 0.03 / 0.16 |
<!-- /REST -->

Over training, the same runs: the training share snapped / soft / one size up ((3, 3, 2), on the curve's 500-question
sample), medians over the seeds:

<!-- REST_CURVE -->
| trained at (3, 2, 1); one size up is (3, 3, 2) | step 1,000 | step 10,000 | step 25,000 | step 50,000 |
|---|---|---|---|---|
| Adam, eps 1e-4 (2 seeds) | 0.41 / 0.89 / 0.07 | 0.41 / 0.79 / 0.05 | 0.27 / 0.35 / 0.07 | 0.20 / 0.29 / 0.05 |
| AdamW, eps 1e-4, weight decay 0.3 (2 seeds) | 0.39 / 0.93 / 0.06 | 0.34 / 0.94 / 0.05 | 0.37 / 0.86 / 0.05 | 0.33 / 0.89 / 0.05 |
| plain SGD, learning rate 0.01 (1 seed) | 0.18 / 0.18 / 0.06 | 0.29 / 0.51 / 0.04 | 0.41 / 1.00 / 0.05 | 0.40 / 1.00 / 0.08 |
| plain SGD 0.01 + weight decay (3 seeds) | 0.18 / 0.18 / 0.07 | 0.29 / 0.47 / 0.05 | 0.21 / 0.59 / 0.04 | 0.17 / 0.52 / 0.03 |
| plain SGD, learning rate 0.1 (2 seeds) | 0.46 / 0.55 / 0.06 | 0.38 / 0.66 / 0.04 | 0.15 / 0.15 / 0.06 | 0.15 / 0.15 / 0.06 |
| plain SGD 0.1 + weight decay (1 seed) | 0.10 / 0.33 / 0.06 | 0.48 / 0.71 / 0.04 | 0.44 / 0.96 / 0.05 | 0.42 / 0.80 / 0.04 |
<!-- /REST_CURVE -->

The 13 cut-off runs, each at the step it reached when Modal stopped it, and its last curve point:

<!-- REST_CUT -->
| arm | seed | stopped at step | at its last curve point: training, snapped / soft | one size up (k + 1 / w + 1, k + 1) | its best over the run | weight norm |
|---|---|---|---|---|---|---|
| Adam, eps 1e-4 | 1 | 45,000 | 0.13 / 0.26 | 0.15 / 0.05 | 0.21 / 0.08 | 62 |
| Adam, eps 1e-4 | 2 | 42,500 | 0.37 / 0.76 | 0.10 / 0.02 | 0.21 / 0.08 | 67 |
| AdamW, eps 1e-4, weight decay 0.3 | 2 | 37,500 | 0.28 / 1.00 | 0.14 / 0.05 | 0.19 / 0.07 | 12 |
| AdamW, eps 1e-4, weight decay 0.3 | 3 | 18,000 | 0.40 / 0.87 | 0.19 / 0.08 | 0.20 / 0.08 | 12 |
| plain SGD, learning rate 0.01 | 1 | 46,500 | 0.33 / 1.00 | 0.17 / 0.05 | 0.22 / 0.08 | 26 |
| plain SGD, learning rate 0.01 | 2 | 32,000 | 0.29 / 1.00 | 0.12 / 0.05 | 0.24 / 0.09 | 26 |
| plain SGD, learning rate 0.01 | 3 | 42,000 | 0.31 / 1.00 | 0.08 / 0.05 | 0.18 / 0.08 | 27 |
| plain SGD 0.01 + weight decay | 0 | 44,000 | 0.20 / 0.61 | 0.13 / 0.04 | 0.22 / 0.09 | 7 |
| plain SGD, learning rate 0.1 | 1 | 36,000 | 0.12 / 0.12 | 0.12 / 0.05 | 0.22 / 0.06 | 47 |
| plain SGD, learning rate 0.1 | 3 | 34,000 | 0.18 / 0.18 | 0.18 / 0.06 | 0.22 / 0.08 | 51 |
| plain SGD 0.1 + weight decay | 1 | 37,000 | 0.36 / 0.95 | 0.18 / 0.05 | 0.21 / 0.08 | 12 |
| plain SGD 0.1 + weight decay | 2 | 45,500 | 0.34 / 0.73 | 0.13 / 0.05 | 0.21 / 0.08 | 13 |
| plain SGD 0.1 + weight decay | 3 | 29,500 | 0.55 / 0.99 | 0.16 / 0.05 | 0.24 / 0.10 | 12 |
<!-- /REST_CUT -->

How sharp the states are at the end, as in §9's table (the runs that finished):

<!-- REST_SHARP -->
| rule | weight norm | training, snapped / soft | d | n | b | m | g |
|---|---|---|---|---|---|---|---|
| the compiled rule, output scale 4 | 45 | 1.00 / 1.00 | 0.95 (100%) | 0.95 (100%) | 0.98 (100%) | 0.96 (100%) | 0.98 (100%) |
| the compiled rule, output scale 20 | 198 | 1.00 / 1.00 | 1.00 (100%) | 1.00 (100%) | 1.00 (100%) | 1.00 (100%) | 1.00 (100%) |
| Adam, eps 1e-4 (2 seeds) | 63 | 0.20 / 0.29 | 1.00 (88%) | 1.00 (92%) | 1.00 (97%) | 1.00 (98%) | 1.00 (98%) |
| AdamW, eps 1e-4, weight decay 0.3 (2 seeds) | 12 | 0.33 / 0.89 | 0.80 (33%) | 0.54 (2%) | 0.92 (54%) | 0.64 (4%) | 0.70 (10%) |
| plain SGD, learning rate 0.01 (1 seed) | 27 | 0.40 / 1.00 | 0.99 (73%) | 0.84 (40%) | 1.00 (86%) | 0.87 (44%) | 0.94 (62%) |
| plain SGD 0.01 + weight decay (3 seeds) | 7 | 0.17 / 0.52 | 0.73 (18%) | 0.54 (0%) | 0.85 (37%) | 0.55 (0%) | 0.64 (0%) |
| plain SGD, learning rate 0.1 (2 seeds) | 82 | 0.15 / 0.15 | 1.00 (100%) | 1.00 (100%) | 1.00 (100%) | 1.00 (100%) | 1.00 (100%) |
| plain SGD 0.1 + weight decay (1 seed) | 12 | 0.42 / 0.80 | 0.80 (37%) | 0.74 (21%) | 0.89 (48%) | 0.71 (17%) | 0.86 (42%) |
<!-- /REST_SHARP -->

**What was tried, and how hard** (dead ends):
- Plain SGD's learning rate (`rest_lr`, one seed each): 0.01, 0.03, 0.1, 0.3 and 1 from random for 5,000 steps;
  0.003, 0.01 and 0.02 from the compiled rule for 10,000.
- From random at (3, 2, 1) (`rest1`): Adam with eps 1e-4 and plain SGD at 0.01 and 0.1, each alone and with weight
  decay, 4 seeds each, 50,000 steps (11 runs finished; 13 stopped at steps 18,000–46,500).
- From the compiled rule at (3, 2, 1) (`rest_stab`): the same six arms at output scale 4, and with plain Adam and
  AdamW 0.3 at output scale 20, 4 seeds each, 10,000 steps.
- From the compiled rule at (3, 3, 2) (`rest_stab3`): plain Adam, Adam with eps 1e-4, plain SGD at 0.003 and 0.01, 4
  seeds each, 10,000 steps.
- In all, 104 runs: 29 from random and 75 from the compiled rule.
- Not tried: (3, 3, 2) from random (at (3, 2, 1) no arm came near the rule, and at (3, 3, 2) none of §9's arms fit
  even the soft path in 20,000 steps); at (3, 3, 2), a learning rate below 0.003 or a clipped gradient, which might
  keep SGD from falling; SGD with momentum; a learning rate that falls over the run; other values of eps; training on
  the snapped path (STE).

**Budget notes.**
- The 104 runs cost about $45 on Modal.
- (3, 2, 1) ran on 4-core CPU containers at 0.19–0.31 s a step, SGD as fast as Adam (3.0–4.1 hours for a run of
  50,000 steps); (3, 3, 2) on L4 GPUs, two or three runs to a GPU, at 0.82–0.92 s a step (2.3–2.6 hours for 10,000).
- As in §9, Modal preempted CPU containers and the runs resumed from their checkpoints. One `rest1` run (AdamW with
  eps 1e-4, seed 2) restarted from step 0 instead of resuming at step 10,500: Modal hands a container that has finished
  one job the next, and such a container sees the Volume as it was when it started, so the run did not find its own
  checkpoint. `modal_nca.py` now reloads the Volume before each job (in later launches; this sweep was already
  running).
- At about 07:25 PT on 2026-10-24 the workspace reached its Modal spend limit, and Modal stopped every container and
  refused new ones, with 13 `rest1` runs unfinished (the last `rest_stab3` record was written 5 seconds before its app
  stopped). Their checkpoints are on the Volume: relaunching `rest1` resumes them where they stopped.

## 11. The traps, and the project's fixes, against the bridge

The project's named traps ([OVERVIEW.md](../OVERVIEW.md) §3, [BRIDGE.md](../bridge/BRIDGE.md) §6), as they show up
here:

| trap | in the bridge | in the NCA |
|---|---|---|
| **analog codes** | one MLP for all rules: right on the soft path, wrong snapped (1.00 vs 0.84) | every soft arm taught by answers: soft 1.00, snapped about 0.5, one size up at "always 0" |
| **the bypass** to the prompt's clean digits | free attention: about 37% of cells read an input; the first gradient points 57% of ports at inputs | does not appear: no cell near the answer can see the prompt, so locality removes the competition between sources. Its NCA form, the register copied straight down, would answer R0; runs answer R0 on 5–40% of training questions, around the 21% at which R0 is the true answer anyway |
| **constant sources** | with the bypass removed, connections settle on constant zeros | scratch cells whose digit is the same on every question: a quarter of them in the median GCL run, up to 90%; the bit-level rule taught by lessons stays at the level of a constant answer (0.17–0.20 of training), with 25–95% of its scratch cells constant |
| **identical, interchangeable parts** (symmetric saddles) | not the binding trap: kinds give every cell a fixed meaning | kinds fix the cells too. In the free format the hidden channels never take the machine's roles: the best match of each to N's digit, the borrow, the memo or the gate is at most a 0.4 lift over the majority value (one 0.65), and two of them can carry nearly the same code |
| **lookup arrangements** (the quotient slots' trap) | soft trial slots co-adapt into a lookup | straight-through states reach 0.32–0.98 of training on the snapped path (3 runs fit it exactly at some step), at chance one size up |
| **self-copy, and the blank grid** (new) | a causal mask keeps a cell from reading itself | told only that the machine's settled grid must be a resting state (`selfcopy2`), every rule learns that, and every run then answers 0 on 91–100% of questions: a cell that reads its own state holds it, and with the own state hidden the blank grid is still a second resting state, which the run from blank never leaves |
| **a second resting state** (new) | the causal mask makes the grid a feed-forward circuit | a rule that reads the row below makes rows feed back; a bigger grid settles into another self-consistent pattern |
| **situations never met** (new) | ports are fixed, so a cell sees only what it uses | a third to two thirds of a bigger grid's cells see a neighbourhood training never showed; a dense rule is undefined there, and even the exact rule, trained further on lessons, drifts there (§10) |

The fixes, in their most generic forms:
- **snapping** (straight-through): closes the analog code's soft/hard gap, and raises the snapped training share, but
  what it learns is a lookup;
- **per-rule local checks**: the subtraction cell's lessons make its table exact, as in the bridge, and nothing
  follows. The plumbing (copies, the memo, the gate chain) has no local check: a copy is only checked at the far end;
- **staged lessons** (the cell's lessons first) and **one step at a time** (the one-link grid first; training at
  k = 2, though with the step law off): no effect;
- the NCA's own: hiding the cell's own state (no self-copy), hiding the row below (no feedback between rows, so no
  second resting state of the whole grid), and training on a grid where every situation occurs (almost no never-met
  situations). These three
  turn the trace's training-size fit into extrapolation, and do nothing for the lessons.

The generator-wired arm (§8) brings the bridge's own trap back: with every cell in reach, the heads read the
prompt's clean digits (39–90% of cells), and its fitted wirings fail where a feature of a source never varied in
training. Its fixes are the NCA's: a training grid where every situation occurs.

Compared with the bridge's ladder: the NCA sits with its loosest notches. The bridge learned wherever the wiring was
fixed or chosen from about 10 options, and taught by lessons on cells and sub-grids. The NCA has no wiring to fix:
moving a digit from where it is to where it is needed is a chain of learned copies through every cell in between, the
lessons say nothing about a copy, and the soft search finds a memorizing code first.

## 12. Caveats

- **Tiny sizes, few seeds.** Base 3, two- and three-digit moduli for training, 2 seeds per arm (6 for the causal
  trace at (3, 2, 1), 14 for the certified setting, 4 per arm in §9 and §10, 2 per arm in §9's starts from the
  compiled rule; 11 of §10's 24 runs from random finished), one learning rate per family (§10 scans plain SGD's on one
  seed). "0/2" means no seed of 2, not never.
- **Few optimizers.** Every run before §10 is Adam (AdamW with decay) at its default eps, which does not rest at the
  compiled rule. §10 adds Adam with eps 1e-4 and plain SGD, without momentum and at a fixed learning rate. At (3, 3, 2)
  plain SGD falls off the rule in every seed, and nothing that might keep it on (a smaller learning rate, a clipped
  gradient) was tried.
- **(3, 3, 2) ran on GPUs, (3, 2, 1) on CPUs.** PyTorch lets cuDNN run float32 convolutions in TF32 by default, so the
  (3, 3, 2) runs of §9 and §10 may have trained at lower precision; §10's plain SGD falls only there, and whether
  precision plays a part is untested.
- **The grid is designed.** The rows' kinds (SHIFT, SUB, SEL; SPLIT, MUL), the number of ticks, where the prompt is
  written and where the answer is read are all given, as the bridge gave its notches the grid. Nothing here asks the
  cells to discover the worksheet's layout. A cell knows its own kind and its neighbours' kinds.
- **The compiled format's channels are designed too**, even when only d, n and b are read by any lesson: there are
  exactly as many channels, of exactly the sizes, as the compiled machine uses. The free format keeps the sizes but
  frees the meanings.
- **The cell's lessons state the wiring** (§2): they query the rule at U.d, U.n, L.b. That is more design than the
  bridge's cell lessons carried.
- **The trace is not a lesson the task offers.** It tells every cell its next state at every tick. It is here to
  separate what the format can learn from what the lessons can teach.
- **Budgets are small, except in §9 and §10.** 3,000 steps for the MLP rules in §3–§8; the bit-level arm, which
  usually needs far longer, got 10,000 in its last suite. §9 trains 20,000 steps at (3, 3, 2) and 50,000 at (3, 2, 1),
  without early stopping; §10 50,000 from random and 10,000 from the compiled rule. The first bit-level trace suite
  ran out of memory (8 GB per container) and was stopped; its successor samples 6 ticks of the trace per step. The
  gates' wiring was random and fixed throughout: DiffLogic CA's structured perception stage, or wiring chosen by
  whole-option search, was not tried.
- **The compile check of the two biggest squares, (3, 5) and (5, 3), ran out of memory** (8 GB per container) on
  every try and was stopped; the compile table has every other size. (The runs that were still going when §3–§8
  were first written have since been folded in.)
- **The generator-wired arm's description of a pair is designed**: row and column offsets, both kinds, edge distances,
  chosen so that the compiled wiring is expressible and nothing depends on the grid's size. Its heads have fixed roles
  only through the cell's lessons, which query head A as the minuend, B as N's digit and C as the borrow; the trace it
  learns from is the bridge's machine, with digit-and-borrow cells. Its seeds: 2 per lesson arm, 12 per trace arm at
  (3, 2, 1).
- **"Certified" is the bridge's tiny test**: exact on the training size and on two bigger sizes. The bridge sampled
  the bigger sizes to 20,000; here the suites trained at (3, 2, 2) and (3, 3, 2) sampled 2,000–5,000, and `cover2`
  and `cover3` checked every question, which is where the rare misses of §4 showed up. Sampled certificates can hide
  misses of that size (a 0.03% miss rate escapes a sample of 3,000 about half the time). It is not H1's
  certification, and nothing here was run on H1.

## 13. Reproduce

Run from `cpu_first/` with the sotaku venv's Python (`~/github/sotaku/venv/bin/python`, torch 2.8.0). Local commands
take seconds to a minute and stay under 1 GB; the sweeps run on Modal (app and Volume `nca`, launched through a remote
driver so that the sweep does not depend on the laptop staying connected; at most 32 CPU containers and 11 L4 GPUs per
launch, the GPU runs a few to a GPU, each in its own process).

The compile check, every size (about 45 minutes locally), or from the per-size Modal jobs:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite compile
```

```bash
~/github/sotaku/venv/bin/python -m modal run nca/modal_nca.py --suite compile --fetch
```

```bash
~/github/sotaku/venv/bin/python -m nca.compile assemble
```

The sweeps (`suites.py`: explore, ladder2, ladder3, free2, sparse1, causal2, partial1, cover1, cover2, square1,
logic3, diag1), each launched and fetched the same way:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite explore
```

```bash
~/github/sotaku/venv/bin/python -m modal run nca/modal_nca.py --suite explore --fetch
```

The check of the certified rules further out, from the weights cover2 kept (suite `verify`, one job per record), then
collected into `records/verify.json`:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite verify
```

```bash
~/github/sotaku/venv/bin/python -m nca.verify
```

Where a saved rule misses (a few minutes locally, evaluation only):

```bash
~/github/sotaku/venv/bin/python -m nca.where cover2 w3_causal_trace_s7 3 4 3
```

The generator-wired arm's compile check (seconds), its suites (`hyper1`, `hyper2`, `hyper3`, `hyper_stab`,
`hyper_diag`, launched and fetched as above):

```bash
~/github/sotaku/venv/bin/python -m nca.hyper
```

Trimming and agreement (§9): its suite, then a look at every run's latest curve point while it trains (each run saves
itself every 500 steps and resumes from there when its container is restarted), then the records:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite trim1
```

```bash
~/github/sotaku/venv/bin/python -m modal run nca/modal_nca.py --suite trim1 --peek
```

```bash
~/github/sotaku/venv/bin/python -m modal run nca/modal_nca.py --suite trim1 --fetch
```

The AND-mask across columns and the start at the compiled rule, launched, watched and fetched the same way:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite trim_cols,trim_stab
```

```bash
~/github/sotaku/venv/bin/python -m modal run nca/modal_nca.py --suite trim_cols,trim_stab --fetch
```

The run that stopped without a record (§9's budget notes): its last checkpoint's curve, read from the Volume before
it was resumed (seconds, written to `records/trim_cut.json`; now that every run has a record, a rerun keeps the file
as it is), then its resume from that checkpoint, fetched as above:

```bash
~/github/sotaku/venv/bin/python -m nca.cut
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite trim1 --only w2_wd0.1_s1
```

The weight norm the compiled rule needs and the sharpness of the trained runs' states (seconds, written to
`records/sharpness.json`), and whether Adam rests at the compiled rule (under a minute):

```bash
~/github/sotaku/venv/bin/python -m nca.sharpness
```

```bash
~/github/sotaku/venv/bin/python -m nca.rest
```

The optimizers that can rest at the rule (§10): plain SGD's learning rate, then the arms from random and from the
compiled rule at (3, 2, 1), then the start at (3, 3, 2) on GPU, each launched, watched and fetched as above:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite rest_lr
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite rest1,rest_stab
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite rest_stab3
```

The curves of the `rest1` runs that Modal stopped at the workspace's spend limit, read from the Volume (seconds,
written to `records/rest_cut.json`); launching `rest1` again resumes them from their checkpoints:

```bash
~/github/sotaku/venv/bin/python -m nca.cut rest
```

Where the drifted rule goes wrong (about 30 seconds, written to `records/erosion.json`), then the lessons' gradient at
the compiled rule, the rule shrunk as weight decay alone would shrink it, and plain SGD at the rule where every lesson
is met (seconds each):

```bash
~/github/sotaku/venv/bin/python -m nca.erosion
```

```bash
~/github/sotaku/venv/bin/python -m nca.rest grads
```

```bash
~/github/sotaku/venv/bin/python -m nca.rest shrink
```

```bash
~/github/sotaku/venv/bin/python -m nca.rest sgd
```

The situations count (seconds) and the tables:

```bash
~/github/sotaku/venv/bin/python -m nca.situations
```

```bash
~/github/sotaku/venv/bin/python -m nca.fill
```

Files:
- `grid.py`: the worksheet layouts (division and square), the data, the machine's reference values, the compiled
  rule's clauses;
- `model.py`: the NCA (the 3 × 3 rule as an MLP or as logic gates, the neighbourhoods, the long-range reaches, the
  tick loop, soft / straight-through / hard states);
- `logic.py`: the learned logic-gate rule;
- `hyper.py`: the generator-wired NCA (all-to-all heads weighted by g), its compiled point and check
  (`records/compile_hyper.json`), the wiring probes, the one-port census and the first pull;
- `compile.py`: the compile check;
- `train.py`: the lessons, training (Adam, AdamW or plain SGD; the AND-mask and delta-bar-delta speeds; checkpoints),
  certification, the probes (the plumbing probe among them) and the per-cell diagnosis;
- `suites.py`, `modal_nca.py`: the sweeps and their launcher;
- `situations.py`: how many of a bigger grid's local situations are new;
- `verify.py`: the certified rules checked further out from their saved weights;
- `where.py`: where a saved rule misses on every question of a size (its records: `records/where_*.json`);
- `sharpness.py`: the weight norm the compiled rule needs on the soft path, and how sharp the trim and rest1 runs'
  states are;
- `rest.py`: whether an optimizer rests at the compiled rule (Adam, Adam with eps 1e-4, plain SGD), the lessons'
  gradient there, and the rule shrunk as weight decay alone would shrink it;
- `erosion.py`: where a rule that kept its training answers goes wrong one size up;
- `cut.py`: the curves of runs that stopped without a record (§9's trim run, saved before it was resumed; §10's
  rest1 runs that Modal stopped);
- `summarize.py`, `fill.py`: the tables;
- `records/`: compile.json, situations.json, verify.json, where_*.json, sharpness.json, trim_cut.json, rest_cut.json,
  erosion.json; `runs/`: one jsonl per suite, each record holding its config, its training log and its probes (and,
  in cover2, cover3, the trim suites and the rest suites but rest_lr, the trained rule's weights; in the trim and rest
  suites also its curve).
