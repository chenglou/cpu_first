# The handwired bridge: the digit machine as a handwired program, loosened

## In plain words

**The question.** The attention bridge ([BRIDGE.md](../bridge/BRIDGE.md)) showed that our long-division calculator is one
particular setting of a generic looped transformer, and asked how far that network can be loosened before training
from random stops finding the calculator. Its answer: the digit tables can be learned, the wiring only from short
menus, and free attention never. The first thing that went wrong was a "bypass": attention preferred the clean prompt
digits to the computed cells. Attention picks a source with a softmax, so the candidate sources compete. Was that
competition the problem?

handwired, the user's other project, compiles ordinary JavaScript into a network of just two operations: weighted
sums, and "a value times a switch" (ReGLU, a value multiplied by max(0, gate)). There is no attention anywhere. This
document writes the calculator in handwired, compiles it, and loosens the compiled network the same way.

**Step 1. The calculator is a handwired program.** One "tick" of the long-division worksheet updates every cell from the
cells it reads:
- a table lookup for the subtraction, written the handwired way: multiply the one-hot digits together, then sum the
  table's 0/1 entries;
- a select for the keep-or-discard gate;
- a copy for the shift.

handwired compiles that tick into 3 layers, 259 operations at the smallest size, and a loop runs it 9 to 42 times. It
gets every answer and every intermediate cell exactly right at 14 sizes, in handwired's own evaluators and loaded into
PyTorch. In this network the wiring is not attention: it is which weights are nonzero.

**Step 2. Loosen it part by part, and train from random.** Each part was trained with the attention bridge's lessons
and judged by its certification:
- **The table learns,** exactly as in the attention bridge: with the cell's own lessons, every seed at the smallest
  size and 10 of 12 one size up.
- **The compiled "switches" that build the table's inputs do not.** Freeing the numbers in the compiled product circuit,
  while keeping which inputs each product reads, certified in none of 40 runs. The reason is concrete: each product is
  the only unit that can carry its combination of digits, and 3 in 8 of them start switched off for good (a ReLU gate
  below zero on every input never gets a gradient).
- **A generic ReGLU network over the same inputs learns, though less reliably** than the attention bridge's ReLU
  network (31 of 32): 12 of 16 runs at the smallest size, 1 of 16 one size up, where a softer reading of its outputs
  brings it back to 6 of 7.
- **The wiring learns only from short menus.** With the one-link law (the small check the attention bridge used), the
  limit depends on the switchboard:
  - a normalized ReLU switchboard: up to 7 options per connection;
  - softmax: 10, exactly as in the attention bridge;
  - softplus, a ReLU whose switches can never switch off for good: 13 in every seed (though only 2 of 4 seeds at 10);
  - switches with no competition at all: none, not even 7;
  - free wiring, where any cell may be read: never, as free attention never did.
- **Untying the loop,** one set of weights per tick, breaks every part that learned tied: the table 1 of 16, the 7-option
  menus 0 of 16.

**Step 3. Compared with attention.**
- **The bypass does not go away.** At the start, the answers' gradient points a connection at the clean prompt in 57% of
  the cells, for every kind of switchboard, exactly the share the attention bridge measured for free attention. At the
  end a third of the free connections still read the prompt. The pull comes from the data, not from softmax.
- **Removing the competition makes it worse, and shows why softmax works where it does.** Without competition, the true
  source's switch is pushed off in the first steps and never comes back. With a normalized ReLU it dies once a wrong
  source has won. Softmax recovers from the same early mistakes because its weights never reach zero. Where softmax fails
  (13 options) it freezes its wrong choices in a separate valley. The ReLU failures sit on flat ground, with the right
  answer in a straight line but the switches that would have to move dead.
- **The other traps:**
  - the near miss from the answers alone is the attention bridge's own: the same 42%, the same 4 of 7 connections;
  - a constant source wins without competition (36% of connections read a constant zero);
  - one new impostor: a generic cell that may read entries the lessons never vary learns a rule that depends on them;
  - no renamed digits.

**Over-parameterization and projection: the user's idea.** The idea: exact weights transcribed from a program are too
compressed to train, and a bigger, redundant version of the same program might train where the compact one does not.
- **Spare units help a little near the solution:** with every compiled unit in 4 copies, the product circuit recovers
  from moderate noise in 3 of 4 runs against 1 of 4.
- **From random, spare units help only at the smallest size:** the 4-copy circuit, also learning the table's pattern,
  certifies in 6 of 8 there, in none one size up, and does not learn H1.
- **Re-expressing the program through a random projection into a bigger space never helps.** A fixed linear lift adds no
  units: it only changes the optimizer's geometry, and the noise it adds lands where the compiled weights were zero.
- **Exploring wide and committing narrow does not help either:** neither alternating wide training with snapping back to
  an exact narrow program, nor keeping the best of a small population of narrow networks.

**The shared finding.** handwired's lab found that table-like parts of a compiled program train and exact parts often do
not; the attention bridge found that tables train and routing does not. Here the line runs inside the compiled program
itself:
- **Its table rows train:** one independent weight per entry, each correctable on its own.
- **Its switches do not:** both the wiring between cells and the compiled products, which route each digit combination
  to its own unit. Each switch is a discrete choice carried by one parameter, and a wrong or dead setting blocks its own
  gradient.

Spare carriers (copies, a dense cell) remove the dead-switch cause at the smallest size. They do not remove the harder
problem behind it: a soft search that must commit many discrete choices at once.

**H1.** The certified wide-menu machine, with its two digit tables replaced by generic ReGLU networks in the shape of
handwired's compiled cell, **did not certify** on the official H100 manifest. Its seed, 74, fails deterministically, and
does so on a CPU too. The same file **certified through the official evaluator on a CPU with seed 1** (1 of 4 seeds):
T = 64 on both ladders, every rung exact. So handwired's cell can carry the whole H1 machine, but not reliably, as at
the tiny size. handwired's own compiled circuit, widened 4 times, did not learn H1 in a training diagnostic, and got no
official run.

## Contents

1. The program and the compile
2. The trainable network
3. The notches, trained from random
4. The routing without softmax
5. Diagnosis: the compiled program as a probe
6. Near the compiled point: Net2Net widening and projection
7. H1
8. Caveats
9. Reproduce

## 1. The program and the compile

[`js/machine.ts`](js/machine.ts) writes the attention bridge's tiny machine ([BRIDGE.md](../bridge/BRIDGE.md) §1) as a
handwired program, with handwired's own `Circuit` and `cells.ts` (read-only imports from `~/github/handwired`; nothing
there was changed). The grid is the same: base B, w-digit moduli, dividends v < N·B^k, one cell per table application,
rows N, R0, then per step a SHIFT row and per link a SUB row and a SEL row. Every cell holds a one-hot digit (B exact
bits) and a one-hot borrow (2 exact bits).

The program is one **tick**: it computes every scratch cell's next state from the current states of the cells it reads
(its ports A, B and C, the same ports as the attention bridge's heads). The **host loop** runs it `depth` times, writing
the state cells back after each evaluation, which is how handwired runs any program with state
(handwired's `docs/tutorial.md`, in `~/github/handwired`, outside this repo). Input cells (N's and v's digits, the constant zeros) are input
cells: the host writes them and the program keeps them. In handwired's idioms:

| part | JavaScript it replaces | handwired operations |
|---|---|---|
| SUB, the table | `x = a - b - c; digit = (x + B) % B; borrow = x < 0` | the joint one-hot of (a, b, c) as products, `multiply(multiply(a_i, b_j), c_k)` (two ReGLU layers, B² and 2B² units), then one affine row per output bit whose 0/1 coefficients are the table ("a value that depends on a one-hot is one sum over the one-hot", patterns.md) |
| SEL, the commit gate | `digit = c ? a : b` | `emitSelect`: c × a + (1 − c) × b, ReGLU products |
| SHIFT, the copy | `digit = a` | a one-term affine kept as its own operation (`roundAffine`), so that the copy is a weight and not a host-side alias |
| the wiring | which cell each port reads | which wires each operation reads: the sparsity pattern of the first layer's weights |
| the loop | the ticks | the host loop: one compiled program evaluated `depth` times (the same weights every tick) |

Nothing else: no index reads, no boolean atoms, no gathers (the one-hot groups are declared to the validator only, so
the compiler keeps masked sums dense). `compileTensorProgram` turns the tick into **3 layers of sparse affine rows, some
of them gated by a ReLU (ReGLU)**, each layer reading any earlier value. At (3, 2, 1) that is 259 live operations, 198 of
them ReGLU products, and 659 stored weights.

[`js/compile.ts`](js/compile.ts) compiles every size of the attention bridge's compile table and checks it in
handwired's own evaluators: the compiled tensor executor on every question (a random 20,000 for the three largest), the
reference evaluator `evaluateReference` (the definition of what a program computes) on up to 300, and the tensor
executor again from random one-hot scratch states on up to 2,000. It writes each compiled program to
`records/compiled/B{B}w{w}k{k}.json`: the layers, every grid cell's wires, the next value of every source cell and the
live operations with tags (which cell, which role).

[`load.py`](load.py) runs those layers in PyTorch as they are, knowing nothing of what they compute, inside the same
host loop, and checks (record `records/compile.json`):
- the JS layout is the attention bridge's `grid.Layout` (kinds, input wiring, ports, answer cells, ticks);
- every question comes out exactly v mod N, in float64 and in float32;
- every cell of the grid holds the machine's own value (`grid.machine_reference`), digit and borrow;
- the same from random one-hot scratch states.

<!-- COMPILE -->
| (B, w, k) | questions | cells | ticks | live operations | layers | ReGLU gates | stored weights | handwired's evaluators (tensor / reference / random start) | PyTorch, every answer and cell, f64 and f32, also from random scratch |
|---|---|---|---|---|---|---|---|---|---|
| (3,2,1) | 99 | 28 | 9 | 259 | 3 | 198 | 659 | 99/99, 99/99, 99/99 | exact |
| (3,2,2) | 297 | 48 | 18 | 516 | 3 | 396 | 1,318 | 297/297, 297/297, 297/297 | exact |
| (3,2,3) | 891 | 68 | 27 | 773 | 3 | 594 | 1,977 | 891/891, 300/300, 891/891 | exact |
| (3,2,4) | 2,673 | 88 | 36 | 1,030 | 3 | 792 | 2,636 | 2673/2673, 300/300, 2000/2000 | exact |
| (3,3,2) | 2,835 | 60 | 22 | 686 | 3 | 528 | 1,756 | 2835/2835, 300/300, 2000/2000 | exact |
| (3,3,3) | 8,505 | 85 | 33 | 1,028 | 3 | 792 | 2,634 | 8505/8505, 300/300, 2000/2000 | exact |
| (2,3,3) | 176 | 55 | 18 | 293 | 3 | 192 | 651 | 176/176, 176/176, 176/176 | exact |
| (2,4,4) | 1,472 | 84 | 28 | 486 | 3 | 320 | 1,084 | 1472/1472, 300/300, 1472/1472 | exact |
| (4,2,2) | 1,824 | 64 | 26 | 1,220 | 3 | 1,008 | 3,342 | 1824/1824, 300/300, 1824/1824 | exact |
| (4,2,3) | 7,296 | 92 | 39 | 1,829 | 3 | 1,512 | 5,013 | 7296/7296, 300/300, 2000/2000 | exact |
| (5,2,2) | 7,250 | 80 | 34 | 2,368 | 3 | 2,040 | 6,758 | 7250/7250, 300/300, 2000/2000 | exact |
| (4,3,2) | 20,000 (sample) | 80 | 32 | 1,624 | 3 | 1,344 | 4,454 | 20000/20000, 300/300, 2000/2000 | exact |
| (5,3,2) | 20,000 (sample) | 100 | 42 | 3,154 | 3 | 2,720 | 9,008 | 20000/20000, 300/300, 2000/2000 | exact |
| (3,4,3) | 20,000 (sample) | 102 | 39 | 1,283 | 3 | 990 | 3,291 | 20000/20000, 300/300, 2000/2000 | exact |
<!-- /COMPILE -->

The compiled program grows linearly with the grid: at base 3 one SUB cell is 9 + 18 products and 5 output rows, one SEL
cell 6 products, 3 sums and a complement shared by its row, one SHIFT cell 3 copies. A link of P places holds
P (3B² + 2B) ReGLU products; at (5, 3, 2), 100 cells and 42 ticks, the tick is 3,154 operations.

## 2. The trainable network

[`hnet.py`](hnet.py) reads the compiled program back and regroups its weights by what they do, so that each part can be
freed on its own:
- **cell circuits.** Every scratch cell of a kind does the same work on its ports ("the same work on every item",
  patterns.md); handwired builds one copy per cell. `extract` reads each kind's circuit off the compiled operations (a
  list of affine and ReGLU units in layers over the cell's 3 × (B + 2) port entries) and checks that every copy, at
  every size, is the same circuit.
- **the routing.** Which cells a port reads, as a separate matrix R_h [cells × cells] (compiled: one-hot rows), so that
  the sparsity pattern can be loosened.
- **the loop.** One block of these weights every tick, or one per tick (untied).

At its compiled point this network equals handwired's layers exactly (largest difference 0.0 on every cell, at (3, 2, 1),
(3, 2, 2), (4, 2, 2), (3, 3, 2) and (5, 2, 2); `hnet.check_against_program`).

**What a soft tick writes back.** Handwired's program is exact on exact bits. Trained, its states are soft, and the
product cells are polynomials of degree 3 per tick, composed over 9 to 42 ticks: a port whose weights add up to 1.05
instead of 1 grows its cell's mass by about 1.16 a tick, and the answer's mass by that to the power of the circuit's
degree. With an unnormalized multiplexer the first pilot went to NaN within 50 steps. So the host decides what it writes
back after a soft tick, the soft counterpart of the hard path's snap (`HCfg.writeback`):
- `raw`: the values as computed (the compiled program's own semantics; fine while every part keeps distributions,
  e.g. tables with softmax rows and normalized multiplexers);
- `norm`: each digit group clamped at 0 and normalized;
- `softmax`: softmax(10 × values) per group, for cells whose learned weights can make any value (at the compiled
  point a 0/1 digit comes back within (B − 1) e⁻¹⁰, 10⁻⁴ at base 3, of exact).
The lessons read a digit group the way the write-back does. The hard path, on which every exactness number is
measured, snaps every state to its argmax digit after every tick, as in the attention bridge.

**Lessons and certification** are the attention bridge's, through its `bridge.train.Run` ([`train.py`](train.py)
subclasses it and swaps the network): the answers (A); x mod N = x and the step law (G); the one-link law on a sub-grid
of one link (L); the subtraction cell's lessons on constructed ports, the top-part order facts, the two order laws and
bijectivity (C). With untied ticks the cell's lessons apply to every tick's cell. A run trains on every question of its
size, full batch, Adam (lr 0.05 for tables and routing, 0.01 for generic cell weights), float64, at most 3,000 steps,
and stops 100 steps after its hard path first gets every training answer right. It is **certified** if that hard network
is also exact on every question of (B, w, k + 1) and (B, w + 1, k + 1) (sampled to 20,000).

## 3. The notches, trained from random

[`notches.py`](notches.py), [`suites.py`](suites.py) (`ladder`, `bases`), records `runs/ladder.jsonl` and
`runs/bases.jsonl`, 4 seeds. Each notch frees one part and keeps the rest compiled:

Sizes at base 3: the table has 90 learned numbers; the freed product circuit 149 (its 27 products' left and gate
coefficients and biases on the compiled sparsity, and its 5 output rows over the 18 joint products; 194 when every
kind's circuit is freed); the dense ReGLU cell 990; a 7-option menu 126 gates (one per port, query kind and option);
dense routing 4,320.

| notch | freed, from random | kept compiled |
|---|---|---|
| 1 | the subtraction cell's 5 output rows over its 18 joint products (the table), each row a softmax (the attention bridge's table form), or plain affine weights (1') | the products, the select, the copy, the wiring, the loop |
| 2 | every value in the subtraction cell's circuit (products' left and gate coefficients and biases, the output rows' weights): handwired's compressed exact circuit with its sparsity kept, including the output rows' (which joint product feeds which output: the table's pattern); 2' frees every kind's circuit; 2-do makes the output rows dense (the table's pattern learned too); x4: every unit in 4 Net2Net copies (§6) | the circuit's sparsity, the other cells, the wiring |
| 2d | the subtraction cell as a dense ReGLU network over the port entries the compiled cell reads (A's and B's digits, C's borrow): two ReGLU layers of the compiled widths (9 and 18 at base 3), each reading every earlier layer, then an affine layer; 2d' reads every port entry; 2d'' every kind's cell | the other cells, the wiring |
| 2t | the subtraction table as a dense ReGLU network's outputs on every one-hot input, read by the compiled products (the form the H1 file uses, §7) | the products, the wiring |
| 3 | the wiring: each port's sources, as a multiplexer over a menu of relative sources (7 to 20), or over every cell (dense routing, with or without L1); tables given or learned | the cells |
| 4 | the loop: one set of weights per tick | |

<!-- LADDER -->
| notch | lessons | (3,2,1) | (3,2,2) |
|---|---|---|---|
| 1. the subtraction table learned (softmax rows) | AC | 4/4 (50) | 2/4 (75) [0.19] |
| 1. the subtraction table learned (softmax rows) | GC | 4/4 (50) | 4/4 (87) |
| 1. the subtraction table learned (softmax rows) | ACL | 4/4 (50) | 4/4 (50) |
| 1. the subtraction table learned (softmax rows) | A | 0/4 fit1 [0.54] | 0/4 fit1 [0.40] |
| 1. the subtraction table learned (softmax rows) | G | 0/4 fit1 [0.52] | 0/4 [0.35] |
| 1'. the same, plain affine rows | AC | 2/4 (37) [0.19] | 1/4 (25) [0.23] |
| 1'. the same, plain affine rows | GC | 2/4 (50) [0.25] | 2/4 (250) [0.26] |
| 2. the subtraction cell's product circuit, values learned | AC | 0/4 [0.30] | 0/4 [0.12] |
| 2. the subtraction cell's product circuit, values learned | GC | 0/4 [0.31] | 0/4 [0.18] |
| 2. the subtraction cell's product circuit, values learned | GCL | 0/4 [0.31] | 0/4 [0.20] |
| 2. the same, values started at N(0, 1) | GC | 0/4 [0.26] | 0/4 [0.18] |
| 2. the same, values started at N(0, 1) | AC | 0/4 [0.37] | 0/4 [0.18] |
| 2'. every cell's circuit, values learned | GC | 0/4 [0.17] | 0/4 [0.18] |
| 2'. every cell's circuit, values learned | GCL | 0/4 [0.14] | 0/4 [0.19] |
| 2d. the subtraction cell a dense ReGLU network | AC | 3/4 (125) [0.24] | 0/4 [0.31] |
| 2d. the subtraction cell a dense ReGLU network | GC | 3/4 (125) [0.12] | 0/4 [0.20] |
| 2d. the subtraction cell a dense ReGLU network | GC+l1 | 3/4 (125) [0.10] | 0/4 [0.19] |
| 2d. the subtraction cell a dense ReGLU network | ACnb | 3/4 (125) [0.21] | 1/4 (2500) [0.11] |
| 2d'. the same, reading every port entry | AC | 1/4 (225) fit1 [0.98] | 0/4 [0.18] |
| 2d'. the same, reading every port entry | GC | 1/4 (650) [0.48] | 0/4 [0.18] |
| 2d''. every cell a dense ReGLU network | GC | 0/4 [0.23] | 0/4 [0.18] |
| 2d''. every cell a dense ReGLU network | GCL | 0/4 [0.15] | 0/4 [0.19] |
| 3. 7-option menu (softmax), tables given | A | 0/4 [0.39] | 0/4 [0.30] |
| 3. 7-option menu (softmax), tables given | AL | 4/4 (100) | 4/4 (112) |
| 3. 7-option menu (normalized ReLU), tables given | A | 0/4 [0.36] | 0/4 [0.24] |
| 3. 7-option menu (normalized ReLU), tables given | AL | 4/4 (100) | 4/4 (112) |
| 3. 7-option menu (ReLU, no competition), tables given | A | 0/4 [0.18] | 0/4 [0.00] |
| 3. 7-option menu (ReLU, no competition), tables given | AL | 0/4 [0.11] | 0/4 [0.00] |
| 3. 7-option menu (plain ReLU multiplexer, host normalizes), tables given | AL | 4/4 (50) | 3/4 (50) [0.17] |
| 3. 7-option menu (plain ReLU multiplexer, host normalizes), tables given | A | 0/4 [0.26] | 0/4 [0.18] |
| 3. 7-option menu (softmax), table learned | ACL+f | 3/4 (425) [0.37] | 3/4 (425) [0.19] |
| 3. 7-option menu (softmax), table learned | ACL | 0/4 [0.27] | 0/4 [0.17] |
| 3. 7-option menu (normalized ReLU), table learned | ACL | 0/4 [0.33] | 0/4 [0.17] |
| 3. 7-option menu (normalized ReLU), table learned | ACL+f | 1/4 (375) [0.40] | 1/4 (400) [0.30] |
| 3. 7-option menu (plain ReLU multiplexer, host normalizes), table learned | ACL+f | 3/4 (350) [0.36] | 2/4 (350) [0.19] |
| 3. 7-option menu (plain ReLU multiplexer, host normalizes), table learned | ACL | 0/4 [0.33] | 0/4 [0.17] |
| 3. dense routing (softmax), tables given | A | 0/4 [0.58] | 0/4 [0.33] |
| 3. dense routing (softmax), tables given | AL | 0/4 [0.44] | 0/4 [0.16] |
| 3. dense routing (normalized ReLU), tables given | A | 0/4 [0.25] | 0/4 [0.31] |
| 3. dense routing (normalized ReLU), tables given | AL | 0/4 [0.21] | 0/4 [0.15] |
| 3. dense routing (normalized ReLU), tables given | AL+l1 | 0/4 [0.17] | 0/4 [0.13] |
| 3. dense routing (plain ReLU multiplexer, host normalizes), tables given | AL | 0/4 [0.28] | 0/4 [0.16] |
| 3. dense routing (plain ReLU multiplexer, host normalizes), tables given | AL+l1 | 0/4 [0.19] | 0/4 [0.17] |
| 3. dense routing (normalized ReLU), table learned | ACL+f | 0/4 [0.22] | 0/4 [0.16] |
| 4. untied ticks, the table learned | AC | 1/4 (125) [0.75] | 0/4 [0.73] |
| 4. untied ticks, the table learned | GC | 0/4 [0.92] | 0/4 [0.79] |
| 4. untied ticks, the dense subtraction cell | AC | 0/4 [0.54] | 0/4 [0.30] |
| 4. untied ticks, the dense subtraction cell | GC | 0/4 [0.55] | 0/4 [0.25] |
| 4. untied ticks, 7-option menu (softmax), tables given | AL | 0/4 [0.35] | 0/4 [0.27] |
| 4. untied ticks, 7-option menu (normalized ReLU), tables given | AL | 0/4 [0.35] | 0/4 [0.31] |
<!-- /LADDER -->

What the ladder says:
- **The table (1) learns exactly as in the attention bridge** (its notch 1a: AC 4/4 and 2/4 [0.17], GC 4/4 and 4/4):
  with the cell's lessons, 12 of 12 at (3, 2, 1) and 10 of 12 at (3, 2, 2); from the answers alone, or with the
  network's laws, 0 of 16, though three runs fit every training answer and fail one size up (16 or 17 of 18 table
  digits right). As plain affine rows (1'), whose entries are not kept in a simplex, 7 of 16.
- **The compiled product circuit with its values freed (2) does not learn: 0 of 40**, at both sizes, with any lesson set,
  from either start scale (0 of 16 when every kind's circuit is freed), although it is given more than the table notch:
  its output rows keep the compiled sparsity, so which joint product feeds which output digit (the table's pattern) is
  still given, and only the numbers are learned. Its failure has a simple cause. Each of its 27 products reads one fixed
  pair of input bits, so each is the only unit that can carry its combination; and a product whose ReLU gate reads one
  bit through a random weight and bias is off on both values of the bit with probability 3/8. Counted over 200 random
  starts: 3.35 of the 9 first-layer and 6.74 of the 18 second-layer units are off on every one-hot input (37%, at any
  start scale), and a ReLU gate that is off everywhere never receives a gradient. With 4 copies of every unit
  (Net2Net), 0.15 and 0.28 units lose all their copies (§6).
- **A dense ReGLU cell (2d) learns at the small size and mostly not one size up:** 12 of 16 at (3, 2, 1) (one seed fails in
  every tier, with the soft path far above the snapped one, 0.44-0.67 against 0.10-0.24: an analog code), 1 of 16 at
  (3, 2, 2), where its table stalls at 4 to 14 of 18 digits. The attention bridge's ReLU MLP cell (1d) certified 8 of 8
  at these two sizes. Reading every port entry (2d') it learns 2 of 8 at (3, 2, 1), and one more run fits every
  training answer with a rule that depends on entries the lessons never vary (§5). Every kind a dense cell (2d''): 0 of
  16.
- **Routing (3): the 7-option menu learns with the one-link law for every multiplexer that keeps some competition**
  (softmax and normalized ReLU 8 of 8, the plain ReLU multiplexer 7 of 8), **and not at all without it** (0 of 16, ending
  at 0.00 to 0.18 of answers). From the answers alone: 0 of 32. With the table learned too it needs "tables first", as in
  the attention bridge (softmax 6 of 8 with the routing frozen for 300 steps, 0 of 8 without; the attention bridge had
  8 of 8 and 1 of 8). **Dense routing: 0 of 64**, with every multiplexer, with or without L1, as the attention bridge's
  free attention (0 of 16 with the tables given).
- **Untying the loop (4) breaks every notch it touches:** the learned table 1 of 16 (tied: 14 of 16 with the same
  lessons), the dense cell 0 of 16, the 7-option menus 0 of 16 (tied: 16 of 16). Untied runs often fit most training
  answers (0.73 to 0.92 for the table) and stop there: each tick's weights now see only the cells that settle at that
  tick, so every lesson is split D ways.

## 4. The routing without softmax

In the attention bridge a port picks its source by attention: a softmax over the candidate cells, so the candidates
compete for one unit of weight. In handwired's network the routing is a weight matrix. A menu port is a multiplexer,
port = Σ_o w_o · S[source_o], and the question is how the weights w come from the learned gates g
([`hnet.py`](hnet.py) `HNet.mux`):

| multiplexer | weights | what the gradient of one gate sees |
|---|---|---|
| softmax | exp(g_o) / Σ exp(g) | its source against the port's current blend, times its own share (the attention bridge's) |
| normalized ReLU | relu(g_o) / Σ relu(g) | its source against the blend, not weighted by its share; 0 for a gate below 0 |
| ReLU, no competition | relu(g_o) / (Σ relu(g), held constant in the backward pass) | its own source only; 0 below 0 |
| sigmoid, no competition | sigmoid(g_o) / (Σ sigmoid(g), held constant) | its own source only |
| plain ReLU multiplexer | relu(g_o), unnormalized, the host normalizing what it writes back | as the normalized ReLU through the product cells, its own source in the select |
| normalized softplus | softplus(4 g_o) / Σ softplus(4 g) | as the normalized ReLU, but a gate below 0 keeps a (small) gradient |

With softmax, this network *is* the attention bridge's menu network: on every cell of (3, 2, 2), from random gates,
the two agree to 2.4·10⁻¹³ on the soft path and exactly on the hard path (the compiled product cells compute the
attention bridge's multilinear tables on soft states too). So each row below changes one thing, the multiplexer.

**The width table** (`width` and `width2`, records `runs/width.jsonl`, `runs/width2.jsonl`): the tables given, (3, 2, 1),
4 seeds, 2,000 steps, each port a multiplexer over 7 to 20 relative sources (the attention bridge's menus):

<!-- WIDTH -->
| multiplexer | lessons | 7 options | 10 options | 13 options | 16 options | 20 options |
|---|---|---|---|---|---|---|
| softmax | AL | 4/4 (100) | 4/4 (175) | 0/4 [0.25] | 0/4 [0.19] | 0/4 [0.24] |
| normalized ReLU | AL | 4/4 (100) | 0/4 [0.33] | 0/4 [0.25] | 0/4 [0.35] | 0/4 [0.29] |
| ReLU, no competition | AL | 0/4 [0.11] | 0/4 [0.09] | 0/4 [0.16] | 0/4 [0.18] | 0/4 [0.27] |
| sigmoid, no competition | AL | 0/4 [0.33] | 0/4 [0.33] | 0/4 [0.20] | 0/4 [0.14] | 0/4 [0.28] |
| plain ReLU multiplexer, host normalizes | AL | 4/4 (50) | 0/4 [0.33] | 0/4 [0.18] | 0/4 [0.25] | 0/4 [0.31] |
| normalized softplus (a ReLU that never dies) | AL | 4/4 (100) | 2/4 (187) [0.33] | 4/4 (887) | 0/4 [0.33] | 0/4 [0.24] |
| softmax | A | 0/4 [0.39] | 0/4 [0.41] | 0/4 [0.41] | 0/4 [0.39] | 0/4 [0.28] |
| normalized ReLU | A | 0/4 [0.36] | 0/4 [0.39] | 0/4 [0.33] | 0/4 [0.36] | 0/4 [0.39] |
| ReLU, no competition | A | 0/4 [0.18] | 0/4 [0.16] | 0/4 [0.29] | 0/4 [0.20] | 0/4 [0.39] |
| sigmoid, no competition | A | 0/4 [0.18] | 0/4 [0.33] | 0/4 [0.33] | 0/4 [0.33] | 0/4 [0.27] |
| plain ReLU multiplexer, host normalizes | A | 0/4 [0.26] | 0/4 [0.36] | 0/4 [0.33] | 0/4 [0.27] | 0/4 [0.28] |
| normalized softplus (a ReLU that never dies) | A | 0/4 [0.36] | 0/4 [0.42] | 0/4 [0.42] | 0/4 [0.37] | 0/4 [0.21] |
<!-- /WIDTH -->

**Without softmax the edge moves in for handwired's own ReLU multiplexers, and out for one that cannot switch off.**
Softmax, the attention bridge's own rule, learns 7 and 10 options in every seed (its numbers here reproduce BRIDGE.md §4
seed for seed: 4/4 at 100 and 175 steps, 0/4 at 13 and beyond). The normalized ReLU multiplexer learns 7 and not 10.
Gates with no competition at all learn nothing, not even 7 options with the one-link law. The plain unnormalized
multiplexer learns 7 (in 50 steps, the fastest) and not 10. The normalized softplus multiplexer learns 13 in every seed,
past softmax's edge, though only 2 of 4 seeds at 10. From the answers alone, none learns anything.

**Why, traced** (seed 0, the true option's weight against the strongest other option on each of the 7 ports, every 50
steps):
- **No competition: the true gate dies.** On 3 of the 7 ports the first steps push the true option's gate below 0
  (its source does not yet help: every computed cell is still noise), and a ReLU gate below 0 gets no gradient again.
  The port can never choose its true source. The other gates grow without bound (13 by step 400), since nothing ties
  them to each other.
- **10 options: softmax recovers from a wrong commitment, the ReLU does not.** Both reach the same state at step 100:
  the subtraction's minuend has committed to a wrong source (weight 0.96-1.00), the select's old value is undecided.
  Softmax recovers by step 200 (every port right, exact). The normalized ReLU stays there: the true option's gate went
  below 0 while the wrong option won, and it is dead. exp never reaches 0, so every softmax option keeps a gradient.
- **A ReLU that never dies recovers too.** The normalized softplus multiplexer, softplus(4 g) / Σ, which is a ReLU
  except that a gate below 0 keeps a small gradient, recovers at 10 options exactly as softmax does (exact by step
  200).
- **13 options: softmax freezes its wrong commitments, softplus does not.** By step 400 softmax has two ports
  committed to wrong sources with the true option's weight at 0.00 and the loss flat (0.99): it pushed the losing gates
  to about −5, a share of order e⁻¹⁰, and they stopped moving. The softplus multiplexer leaves its losing gates near
  −2.4, keeps flipping ports to their true sources (2 of them between steps 300 and 400), and certifies at step 425.

So the attention bridge's trap at wide menus is not caused by softmax's competition; the competition is what lets a
soft search choose at all. What handwired's own multiplexer adds is a new trap: a ReLU switch that turns off for good.

**The softplus multiplexer over 4 seeds** (`width2`): 7 options 4 of 4, 10 options 2 of 4 (both failures with the
select's two ports on wrong sources), **13 options 4 of 4** (median 887 steps), 16 and 20 options none. It is the only
multiplexer here that learns past 10 options, and it does not get past 13.

**A longer budget** (`width2`, 6,000 steps instead of 2,000, the one-link law, tables given): softmax still learns
nothing at 13, 16 or 20 options, so its edge is not a budget; the softplus multiplexer certifies 13 options as before
and still nothing beyond:

<!-- WIDTH6K -->
| multiplexer | 13 options | 16 options | 20 options |
|---|---|---|---|
| softmax | 0/4 [0.26] | 0/4 [0.19] | 0/4 [0.27] |
| normalized softplus (a ReLU that never dies) | 4/4 (887) | 0/4 [0.33] | 0/4 [0.24] |
<!-- /WIDTH6K -->

**Dense routing** (every cell a port can read, its weight from a table over (the reading cell's kind, the read cell's
kind, row offset, column offset, top place), with or without L1 pressure): see the ladder (§3).

## 5. Diagnosis: the compiled program as a probe

[`probe.py`](probe.py), suite `diagnose` (records `runs/diagnose.jsonl`), (3, 2, 1), seeds 0 and 1, 2,000 steps. The
probes are the attention bridge's (BRIDGE.md §6), with the routing matrix in place of attention:
- **ports**: every port's strongest source per cell against the compiled one: right; a prompt cell the machine does
  not read (a bypass); a constant zero; another computed cell;
- **first gradient**: at the start, which source the answers' gradient on the routing matrix pushes up most;
- **path**: the answer loss on the straight line from the trained weights to the compiled ones;
- for the freed cells, **ignored entries** (the share of outputs that change when the port entries the compiled cell
  never reads are set at random) and **renamed digits** (whether scratch cells hold the machine's digits under a
  consistent renaming: a digit-meaning drift).

<!-- DIAG -->
| case | multiplexer | seed | certified | training, hard / soft | ports right | sources: prompt / zero / other computed | first gradient toward the prompt | loss on the line to the compiled weights | subtraction digits right |
|---|---|---|---|---|---|---|---|---|---|
| dense_answers | plain ReLU multiplexer, host normalizes | 0 | no | 0.40 / 0.36 | 0/7 | 0.52 / 0.07 / 0.29 | 0.57 | 0.78 -> 1.58 -> 0.00 | 18/18 |
| dense_answers | plain ReLU multiplexer, host normalizes | 1 | no | 0.47 / 0.42 | 1/7 | 0.38 / 0.14 / 0.29 | 0.57 | 0.70 -> 0.92 -> 0.00 | 18/18 |
| dense_answers | softmax | 0 | no | 0.54 / 0.54 | 0/7 | 0.43 / 0.07 / 0.40 | 0.57 | 0.60 -> 1.24 -> 0.00 | 18/18 |
| dense_answers | softmax | 1 | no | 0.53 / 0.49 | 0/7 | 0.36 / 0.00 / 0.43 | 0.57 | 0.48 -> 0.93 -> 0.00 | 18/18 |
| dense_link | plain ReLU multiplexer, host normalizes | 0 | no | 0.26 / 0.32 | 2/7 | 0.26 / 0.02 / 0.40 | 0.57 | 0.81 -> 3.94 -> 0.00 | 18/18 |
| dense_link | plain ReLU multiplexer, host normalizes | 1 | no | 0.29 / 0.30 | 0/7 | 0.45 / 0.10 / 0.33 | 0.57 | 0.94 -> 0.97 -> 0.00 | 18/18 |
| dense_link | normalized ReLU | 0 | no | 0.07 / 0.18 | 1/7 | 0.31 / 0.14 / 0.36 | 0.57 | 1.10 -> 2.77 -> 0.00 | 18/18 |
| dense_link | normalized ReLU | 1 | no | 0.15 / 0.18 | 0/7 | 0.31 / 0.12 / 0.40 | 0.57 | 1.10 -> 1.43 -> 0.00 | 18/18 |
| dense_link | ReLU, no competition | 0 | no | 0.18 / 0.18 | 1/7 | 0.02 / 0.14 / 0.52 | 0.57 | 1.10 -> 2.26 -> 0.00 | 18/18 |
| dense_link | ReLU, no competition | 1 | no | 0.17 / 0.22 | 0/7 | 0.31 / 0.14 / 0.40 | 0.57 | 1.06 -> 1.12 -> 0.00 | 18/18 |
| dense_link | softmax | 0 | no | 0.52 / 0.45 | 1/7 | 0.33 / 0.05 / 0.36 | 0.57 | 0.67 -> 1.03 -> 0.00 | 18/18 |
| dense_link | softmax | 1 | no | 0.41 / 0.43 | 1/7 | 0.29 / 0.07 / 0.38 | 0.57 | 0.68 -> 1.00 -> 0.00 | 18/18 |
| menu10_link | plain ReLU multiplexer, host normalizes | 0 | no | 0.18 / 0.36 | 5/7 | 0.07 / 0.02 / 0.14 | 0.29 | 0.89 -> 0.92 -> 0.00 | 18/18 |
| menu10_link | plain ReLU multiplexer, host normalizes | 1 | no | 0.33 / 0.35 | 4/7 | 0.17 / 0.07 / 0.19 | 0.29 | 1.05 -> 1.05 -> 0.00 | 18/18 |
| menu10_link | normalized ReLU | 0 | no | 0.33 / 0.37 | 5/7 | 0.07 / 0.02 / 0.19 | 0.29 | 0.93 -> 0.94 -> 0.00 | 18/18 |
| menu10_link | normalized ReLU | 1 | no | 0.33 / 0.16 | 3/7 | 0.17 / 0.12 / 0.29 | 0.29 | 1.05 -> 1.05 -> 0.00 | 18/18 |
| menu10_link | softmax | 0 | yes | 1.00 / 1.00 | 7/7 | 0.00 / 0.00 / 0.00 | 0.29 | 0.01 -> 0.01 -> 0.00 | 18/18 |
| menu10_link | softmax | 1 | yes | 1.00 / 1.00 | 7/7 | 0.00 / 0.00 / 0.00 | 0.29 | 0.01 -> 0.01 -> 0.00 | 18/18 |
| menu10_link | normalized softplus (a ReLU that never dies) | 0 | yes | 1.00 / 1.00 | 7/7 | 0.00 / 0.00 / 0.00 | 0.29 | 0.00 -> 0.00 -> 0.00 | 18/18 |
| menu10_link | normalized softplus (a ReLU that never dies) | 1 | no | 0.33 / 0.37 | 5/7 | 0.10 / 0.05 / 0.14 | 0.29 | 0.91 -> 0.91 -> 0.00 | 18/18 |
| menu13_link | plain ReLU multiplexer, host normalizes | 0 | no | 0.18 / 0.17 | 3/7 | 0.05 / 0.19 / 0.33 | 0.33 | 1.11 -> 1.13 -> 0.00 | 18/18 |
| menu13_link | plain ReLU multiplexer, host normalizes | 1 | no | 0.33 / 0.16 | 4/7 | 0.05 / 0.14 / 0.24 | 0.33 | 1.06 -> 1.06 -> 0.00 | 18/18 |
| menu13_link | normalized ReLU | 0 | no | 0.33 / 0.28 | 2/7 | 0.14 / 0.17 / 0.36 | 0.33 | 0.98 -> 0.98 -> 0.00 | 18/18 |
| menu13_link | normalized ReLU | 1 | no | 0.16 / 0.22 | 2/7 | 0.17 / 0.21 / 0.33 | 0.33 | 0.95 -> 1.00 -> 0.00 | 18/18 |
| menu13_link | softmax | 0 | no | 0.51 / 0.40 | 3/7 | 0.24 / 0.10 / 0.24 | 0.33 | 0.88 -> 1.55 -> 0.00 | 18/18 |
| menu13_link | softmax | 1 | no | 0.33 / 0.20 | 3/7 | 0.05 / 0.21 / 0.31 | 0.33 | 0.98 -> 1.08 -> 0.00 | 18/18 |
| menu13_link | normalized softplus (a ReLU that never dies) | 0 | yes | 1.00 / 1.00 | 7/7 | 0.00 / 0.00 / 0.00 | 0.33 | 0.00 -> 0.00 -> 0.00 | 18/18 |
| menu13_link | normalized softplus (a ReLU that never dies) | 1 | yes | 1.00 / 1.00 | 7/7 | 0.00 / 0.00 / 0.00 | 0.33 | 0.00 -> 0.00 -> 0.00 | 18/18 |
| menu_answers | plain ReLU multiplexer, host normalizes | 0 | no | 0.36 / 0.37 | 4/7 | 0.00 / 0.10 / 0.29 | 0.12 | 0.83 -> 0.94 -> 0.00 | 18/18 |
| menu_answers | plain ReLU multiplexer, host normalizes | 1 | no | 0.33 / 0.14 | 4/7 | 0.10 / 0.10 / 0.24 | 0.12 | 0.96 -> 1.07 -> 0.00 | 18/18 |
| menu_answers | normalized ReLU | 0 | no | 0.36 / 0.37 | 4/7 | 0.00 / 0.10 / 0.29 | 0.12 | 0.83 -> 0.96 -> 0.00 | 18/18 |
| menu_answers | normalized ReLU | 1 | no | 0.42 / 0.42 | 4/7 | 0.10 / 0.10 / 0.24 | 0.12 | 0.80 -> 0.90 -> 0.00 | 18/18 |
| menu_answers | ReLU, no competition | 0 | no | 0.18 / 0.15 | 1/7 | 0.00 / 0.19 / 0.57 | 0.12 | 0.92 -> 0.94 -> 0.00 | 18/18 |
| menu_answers | ReLU, no competition | 1 | no | 0.18 / 0.15 | 2/7 | 0.10 / 0.14 / 0.43 | 0.12 | 0.92 -> 0.94 -> 0.00 | 18/18 |
| menu_answers | softmax | 0 | no | 0.42 / 0.42 | 4/7 | 0.00 / 0.14 / 0.24 | 0.12 | 0.80 -> 2.77 -> 0.00 | 18/18 |
| menu_answers | softmax | 1 | no | 0.42 / 0.42 | 4/7 | 0.10 / 0.10 / 0.24 | 0.12 | 0.80 -> 2.73 -> 0.00 | 18/18 |
| menu_link | ReLU, no competition | 0 | no | 0.09 / 0.18 | 2/7 | 0.14 / 0.12 / 0.45 | 0.12 | 1.10 -> 2.07 -> 0.00 | 18/18 |
| menu_link | ReLU, no competition | 1 | no | 0.33 / 0.29 | 2/7 | 0.19 / 0.19 / 0.33 | 0.12 | 1.00 -> 1.02 -> 0.00 | 18/18 |
| menu_link | sigmoid, no competition | 0 | no | 0.33 / 0.20 | 1/7 | 0.10 / 0.38 / 0.33 | 0.12 | 1.05 -> 1.32 -> 0.00 | 18/18 |
| menu_link | sigmoid, no competition | 1 | no | 0.33 / 0.20 | 1/7 | 0.19 / 0.33 / 0.33 | 0.12 | 1.05 -> 1.33 -> 0.00 | 18/18 |
<!-- /DIAG -->

<!-- CELLS -->
| notch | lessons | seed | certified | training, hard / soft | subtraction digits / borrows right | outputs changed by the entries the compiled cell ignores | scratch digits right, renamed / as is |
|---|---|---|---|---|---|---|---|
| 2d''. every cell a dense ReGLU network | GC | 0 | no | 0.24 / 0.65 | 18/18 of 18 | 0.00 | 0.77 / 0.58 |
| 2d''. every cell a dense ReGLU network | GC | 1 | no | 0.23 / 0.32 | 13/17 of 18 | 0.00 | 0.58 / 0.54 |
| 2d. the subtraction cell a dense ReGLU network | AC | 0 | yes | 1.00 / 1.00 | 18/18 of 18 | 0.00 | 1.00 / 1.00 |
| 2d. the subtraction cell a dense ReGLU network | AC | 1 | yes | 1.00 / 1.00 | 18/18 of 18 | 0.00 | 1.00 / 1.00 |
| 2d'. the same, reading every port entry | AC | 0 | no | 1.00 / 1.00 | 18/18 of 18 | 0.19 | 0.99 / 0.99 |
| 2d'. the same, reading every port entry | AC | 1 | no | 0.98 / 0.98 | 18/18 of 18 | 0.17 | 0.99 / 0.99 |
| 2'. every cell's circuit, values learned | GC | 0 | no | 0.18 / 0.18 | 8/15 of 18 | · | 0.59 / 0.57 |
| 2'. every cell's circuit, values learned | GC | 1 | no | 0.25 / 0.27 | 6/12 of 18 | · | 0.57 / 0.40 |
| 2. the subtraction cell's product circuit, values learned | AC | 0 | no | 0.35 / 0.36 | 6/16 of 18 | · | 0.66 / 0.54 |
| 2. the subtraction cell's product circuit, values learned | AC | 1 | no | 0.41 / 0.52 | 8/15 of 18 | · | 0.65 / 0.61 |
<!-- /CELLS -->

**The attention bridge's traps, here** (its names, BRIDGE.md §6 and OVERVIEW.md §3; shares are means over the two seeds
and the 7 ports):

| trap | attention bridge | handwired bridge |
|---|---|---|
| bypass: the first gradient points a port at the clean prompt | free attention: 57% of cells | dense routing: 57% at (3, 2, 1) and 62% at (3, 2, 2), for every multiplexer alike (it is measured on the routing matrix, before any multiplexer); 7-option menus 12%, 10 and 13 options 29-33% |
| bypass at the end: ports reading the prompt | free attention: 36-38% of sources | dense routing: 31-45% for softmax, the normalized and the plain ReLU multiplexer; 17% without competition |
| the line to the compiled weights rises first (a separate valley) | free attention +0.3 to +0.4 nats, 7-option menu +1.7 to +1.8 | softmax: 7-option menu +1.9 to +2.0 (answers), 13 options +0.1 to +0.7, dense routing +0.3 to +0.6; every dense failure rises (+0.03 to +3.1) |
| a constant source wins | after the bypass fixes: 19-40% of sources | sigmoid gates without competition: 36% of ports read a constant zero; the others 0-19% |
| near-miss wiring from the answers alone | 7-option menu: 42% of answers, 4 of 7 ports right, every seed | the same near miss for softmax (0.42, 4 of 7 ports, the line rising to 2.75); 0.35-0.42 and 4 of 7 for the normalized and plain ReLU multiplexers |
| a switch dies (new) | does not occur: exp never reaches 0 | without competition the true option's gate on 3 of 7 ports is off for good by step 50 (0 of 16 runs); normalized, the true gate dies once a wrong option has won (10 options, 0 of 8). Most of these failures have **no valley**: for the normalized and plain ReLU multiplexers at 10 and 13 options the line to the compiled weights rises by at most 0.05 nats on the way down; the loss is flat along the dead gates, and training stopped. In the product circuit 3 in 8 units start off (§3) |
| impostor table | MLP cell: tied entries (bijectivity fixes them) | the product circuit ends at 6-8 of 18 digits with its dead units; the dense cell needs bijectivity no more than without it (ACnb 3 of 4 at (3, 2, 1), as AC) |
| a rule that depends on entries the lessons never vary (new) | does not occur: the MLP cell's inputs are only the used entries | a dense cell reading every port entry fits every training answer (0.98-1.00) with 17-19% of its outputs changed by entries the compiled cell ignores; it fails one size up |
| analog code (right on the soft path, wrong snapped) | one MLP for every rule: 1.00 against 0.84 | every cell a dense network: 0.65 soft against 0.24 snapped (seed 0); the dense subtraction cell's failing seed: 0.44-0.67 against 0.10-0.24 |
| digits renamed | free symbols | not found: where every cell is a dense network the scratch cells hold the machine's digits on 54-58% of entries, and the best renaming only raises that to 58-77%, so the cells carry no consistent code |

What the probes say:
- **The bypass is the data's, not softmax's.** The answers' first gradient on the routing matrix is the same for every
  multiplexer, and the trained dense routing ends with a third or more of its sources on the clean prompt, as free
  attention did. Removing softmax's competition does not remove the pull; it removes the search's ability to choose
  (no-competition runs end on other computed cells and constants instead).
- **Two kinds of failure.** Softmax fails by settling in a separate valley: the straight line to the compiled weights
  rises before it falls, as in the attention bridge. ReLU-type multiplexers fail on flat ground: the line falls almost
  all the way, but the gates that would have to move are dead, so their gradient is exactly zero. A softplus gate is a
  ReLU with no flat ground, and it succeeds where softmax freezes (13 options: 7 of 7 ports in both seeds).
- **Freed cells fail in ways the attention bridge's did not show,** because handwired's cells are wider than their port
  format: a cell that may read every port entry learns a rule that the cell's lessons cannot see (they never vary the
  ignored entries), and the compiled product circuit dies unit by unit.
- **No digit renaming,** since the cells' format stays fixed (one-hot digits at fixed places) at every notch here.

## 6. Near the compiled point: Net2Net widening and projection

**The user's idea** (relayed from the parent session; the user's words there): an over-parameterized network lets
gradient descent find decent solutions, so to train exact weights transcribed from JavaScript we might have to make them
"overparametrized, or projected", by projecting the program "in some traditional way", and then a bloated program might
train where a compressed one does not. Two forms, both function-preserving at the start:
- **widened (Net2Net):** every unit of the compiled circuit duplicated 4 times, the weights into the copies split
  (`HCfg.copies`; for routing, 4 copies of every port's gates, the port reading their mean);
- **projected:** the cell re-expressed through a fixed random projection of each port's state into d = 20 or 40
  dimensions (4 or 8 times the state's 5): every weight that reads a port is the projection of a larger trainable
  matrix, every weight that writes the state is read back through the projection's pseudo-inverse (`HCfg.lift`). At the
  compiled start the effective weights equal the compiled ones to 10⁻¹⁵; in the lifted basis they are dense.

A linear projection cannot add anything a network can compute, and under plain gradient descent with an orthonormal
projection it changes nothing at all: the lifted weights' gradient is the projected original gradient. What it can
change is the optimizer's geometry (Adam is not rotation-invariant, and a redundant Gaussian frame acts as a
preconditioner) and the sparsity (the compiled zeros become free numbers). Net2Net widening does add something: spare
units.

**A separate arm, not the user's idea: projected descent** (suite `proj`). Every 25 or 100 steps the learned parts are
snapped onto the nearest exact program (table rows and menu gates to their strongest entry, product weights to the
nearest integer) and training continues from there. This projects *down*, onto exact programs, like the project's
snapping and straight-through paths; it is reported on its own below.

<!-- OVERPARAM -->
**Near the compiled point** (noise sigma on every learned number, then 1,000 steps):

| circuit | lessons | sigma = 0.1 | sigma = 0.3 | sigma = 1 | sigma = 3 |
|---|---|---|---|---|---|
| minimal | GC | 4/4 (4) | 1/4 (0) | 0/4 (0) | 0/4 (0) |
| minimal | the answers | 4/4 (4) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
| widened x4 (Net2Net) | GC | 4/4 (4) | 3/4 (0) | 0/4 (0) | 0/4 (0) |
| widened x4 (Net2Net) | the answers | 4/4 (4) | 1/4 (0) | 0/4 (0) | 0/4 (0) |
| projected, d = 20 | GC | 0/4 (0) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
| projected, d = 20 | the answers | 0/4 (0) | 0/4 (0) | 0/4 (0) | 0/4 (0) |

**From random:**

| circuit | lessons | (3,2,1) | (3,2,2) |
|---|---|---|---|
| table's pattern given, minimal | AC | 0/4 [0.30] | 0/4 [0.12] |
| table's pattern given, minimal | GC | 0/8 [0.31] | 0/8 [0.18] |
| table's pattern given, widened x4 | AC | 2/4 (1475) [0.88] | 0/4 [0.18] |
| table's pattern given, widened x4 | GC | 0/4 [0.92] | 0/4 [0.17] |
| output rows dense, minimal | AC | 0/4 [0.39] | 0/4 [0.16] |
| output rows dense, minimal | GC | 0/4 [0.33] | 0/4 [0.12] |
| output rows dense, widened x4 | AC | 3/4 (725) fit1 [1.00] | 0/4 [0.20] |
| output rows dense, widened x4 | GC | 3/4 (925) [0.92] | 0/4 [0.16] |
| projected, d = 20 | GC | 0/4 [0.41] | 0/4 [0.17] |
| projected, d = 40 | GC | 0/4 [0.33] | 0/4 [0.17] |
<!-- /OVERPARAM -->

The routing's widening (4 copies of every port's gates) near the compiled point (`wstab`; 7-option menus, the tables
given):

<!-- WSTAB -->
| notch | start | lessons | sigma = 0.1 | sigma = 0.3 | sigma = 1 | sigma = 3 |
|---|---|---|---|---|---|---|
| 3. 7-option menu (plain ReLU multiplexer, host normalizes), tables given | minimal | AL | 4/4 (4) | 4/4 (4) | 3/4 (0) | 0/4 (0) |
| 3. 7-option menu (plain ReLU multiplexer, host normalizes), tables given | widened x4 | AL | 4/4 (4) | 4/4 (4) | 1/4 (1) | 1/4 (0) |
| 3. 7-option menu (plain ReLU multiplexer, host normalizes), tables given | minimal | the answers | 4/4 (4) | 4/4 (4) | 0/4 (0) | 0/4 (0) |
| 3. 7-option menu (plain ReLU multiplexer, host normalizes), tables given | widened x4 | the answers | 4/4 (4) | 4/4 (4) | 2/4 (1) | 2/4 (0) |
| 3. 7-option menu (softmax), tables given | minimal | AL | 4/4 (4) | 4/4 (4) | 4/4 (4) | 4/4 (2) |
| 3. 7-option menu (softmax), tables given | widened x4 | AL | 4/4 (4) | 4/4 (4) | 4/4 (4) | 4/4 (4) |
| 3. 7-option menu (softmax), tables given | minimal | the answers | 4/4 (4) | 4/4 (4) | 4/4 (4) | 4/4 (2) |
| 3. 7-option menu (softmax), tables given | widened x4 | the answers | 4/4 (4) | 4/4 (4) | 4/4 (4) | 4/4 (4) |
<!-- /WSTAB -->

From random (`wlearn`; minimal against 4 copies of every unit or every port's gates):

<!-- WLEARN -->
| notch | lessons | 3,2,1 | 3,2,2 |
|---|---|---|---|
| 2. the subtraction cell's product circuit, values learned | GC min | 0/4 [0.31] | 0/4 [0.18] |
| 2. the subtraction cell's product circuit, values learned | GC wide | 0/4 [0.92] | 0/4 [0.17] |
| 3. 13-option menu (softmax), tables given | AL min | 0/4 [0.25] | 0/4 [0.18] |
| 3. 13-option menu (softmax), tables given | AL wide | 0/4 [0.16] | 0/4 [0.18] |
| 3. 13-option menu (plain ReLU multiplexer, host normalizes), tables given | AL min | 0/4 [0.18] | 0/4 [0.19] |
| 3. 13-option menu (plain ReLU multiplexer, host normalizes), tables given | AL wide | 0/4 [0.15] | 0/4 [0.19] |
| 3. dense routing (plain ReLU multiplexer, host normalizes), tables given | AL min | 0/4 [0.28] | 0/4 [0.16] |
| 3. dense routing (plain ReLU multiplexer, host normalizes), tables given | AL wide | 0/4 [0.31] | 0/4 [0.18] |
| 3. dense routing (softmax), tables given | AL min | 0/4 [0.44] | 0/4 [0.16] |
| 3. dense routing (softmax), tables given | AL wide | 0/4 [0.46] | 0/4 [0.20] |
<!-- /WLEARN -->

**Up, then down** (`updown`, [`updown.py`](updown.py); asked by the parent session after the user's question "can't we
project up then down?"): rounds of 200 steps; *blend* trains the 4-copy network, merges it back as the exact inverse of
the split, snaps the narrow machine onto an exact program and checks it, then lifts it again with noise; *pop* trains 4
separate narrow networks, scores each on its own (whole candidates with exact scores, as the wiring tournament does), keeps
the best, snaps and checks it, and clones it with noise. 2,000 steps:

<!-- UPDOWN -->
| notch | lessons | 3,2,1 | 3,2,2 |
|---|---|---|---|
| 2. the subtraction cell's product circuit, values learned | GC up-down blend | 0/4 [0.85] | 0/4 [0.16] |
| 2. the subtraction cell's product circuit, values learned | GC up-down pop | 0/4 [0.44] | 0/4 [0.25] |
| 2-do. the product circuit, output rows dense (the table's pattern learned) | GC up-down blend | 0/4 [0.58] | 0/4 [0.11] |
| 2-do. the product circuit, output rows dense (the table's pattern learned) | GC up-down pop | 0/4 [0.40] | 0/4 [0.19] |
| 3. 13-option menu (softmax), tables given | AL up-down blend | 0/4 [0.33] | 0/4 [0.18] |
| 3. 13-option menu (softmax), tables given | AL up-down pop | 0/4 [0.33] | 0/4 [0.18] |
<!-- /UPDOWN -->

**Projected descent** (`proj`, the snapping reading, not the user's idea): every 25 or 100 steps the learned parts are
snapped onto the nearest exact program; (3, 2, 1), 4 seeds:

<!-- PROJ -->
| notch | lessons | 3,2,1 |
|---|---|---|
| 1. the subtraction table learned (softmax rows) | A every 25 | 0/4 [0.49] |
| 1. the subtraction table learned (softmax rows) | A every 100 | 0/4 fit1 [0.57] |
| 1. the subtraction table learned (softmax rows) | AC every 25 | 1/4 (25) [0.57] |
| 1. the subtraction table learned (softmax rows) | AC every 100 | 4/4 (50) |
| 3. 7-option menu (softmax), tables given | A every 25 | 0/4 [0.34] |
| 3. 7-option menu (softmax), tables given | A every 100 | 0/4 [0.39] |
| 3. 13-option menu (softmax), tables given | AL every 25 | 0/4 [0.18] |
| 3. 13-option menu (softmax), tables given | AL every 100 | 0/4 [0.33] |
| 3. 13-option menu (plain ReLU multiplexer, host normalizes), tables given | AL every 25 | 0/4 [0.32] |
| 3. 13-option menu (plain ReLU multiplexer, host normalizes), tables given | AL every 100 | 0/4 [0.18] |
| 2. the subtraction cell's product circuit, values learned | GC every 25 | 0/4 [0.18] |
| 2. the subtraction cell's product circuit, values learned | GC every 100 | 0/4 [0.18] |
<!-- /PROJ -->

What the arm says:
- **Near the compiled point, spare units help a little; the projection only hurts.** The minimal product circuit
  recovers from noise 0.1 in every run (4 of 4 with the lessons and with the answers alone), from noise 0.3 in 1 of 4
  with the lessons; with 4 copies of every unit, 3 of 4. From noise 1 nothing recovers. The projected circuit is never
  even exact at the start: the noise lands on every port entry in the lifted basis, where the compiled circuit had zeros,
  and no run recovers. The softmax menu is stable at every noise level, minimal or widened. The plain ReLU multiplexer
  recovers from noise 1 in 3 of 4 runs minimal and 1 of 4 widened with the one-link law; widening helps it only when the
  answers alone teach it (2 of 4 against 0 of 4 at noise 1 and 3).
- **From random, widening rescues the compiled circuit only at the smallest size and only if it may also learn the
  table's pattern:** 6 of 8 at (3, 2, 1) with dense output rows and 4 copies (0 of 8 minimal), 0 of 8 at (3, 2, 2); with
  the pattern given, 2 of 16. That is less than it looks: the copies remove the dead-unit cause (2% of units lose all
  their copies, against 37%), but at (3, 2, 2) the widened circuit's table still stalls at 5 to 10 of its 18 digits, on
  both paths (training shares 0.03 to 0.26 snapped, 0.16 to 0.38 soft). It did not carry to H1 either (§7).
- **The routing is not rescued by width,** from random (13-option menus and dense routing: 0 of 32 widened) or by the
  population search over whole narrow networks.
- **The projection does nothing for learning** (0 of 16 from random at d = 20 and 40), as the linear-algebra argument
  above predicts: a fixed linear lift adds no units, so it can only change Adam's geometry and densify the weights.
- **Up-then-down did not help** where it was tried: none of its runs certified, blended or as a population, on the
  product circuit or the 13-option menu, at either size. The menu runs settled on one near miss and stayed there (the
  subtraction's two inputs and the select's new value on wrong sources): every round's snapped narrow machine got the
  same share of the training answers, blended or as the population's best. Snapping in the middle of training (projected
  descent) did not help either: every 25 steps it costs the table notch 3 of its 4 runs, every 100 steps it is neutral,
  and it rescues none of the notches that fail.

So for the user's hypothesis: the compiled weights are indeed hard to learn from random in their compressed form, for a
concrete reason (each unit is irreplaceable, and a third of them start dead), and redundancy in the form of spare units
removes that reason at the smallest size. It does not remove the harder problem behind it, which shows up one size up
and at H1: a soft search that has to commit every one of many discrete choices.

## 7. H1

**The file.** [`h1/submission_wide_reglu.py`](h1/submission_wide_reglu.py) is the attention bridge's certified
[`bridge/h1/submission_wide_mlp.py`](../bridge/h1/submission_wide_mlp.py) (md5 `22bd34d0`, untouched; itself the
certified wide-anchored machine with its two digit tables replaced by generic ReLU MLPs) with those two ReLU MLPs
replaced by generic ReGLU cells in the shape of handwired's compiled subtraction cell:
- `CellReGLU`: two ReGLU layers of 100 and 200 units (B² and 2B² at base 10, the compiled cell's product layers), each
  unit (W_L z + b_L) × relu(W_G z + b_G) reading the one-hot inputs and every earlier layer, then an affine output
  layer; nn.Linear's default start, the output layer scaled by INIT, learned at LR_CELL = 0.01, as the MLP file's;
- the machine reads each cell as a table, the cell's outputs on every one-hot input (the subtractor's 200 and the
  product cell's 100), so the snapped machine at evaluation applies the cell to its digits;
- compiled (`MODE = "designed"`), the cell is handwired's exact circuit: the first layer's unit (i, j) is x_i ×
  relu(y_j), the second layer's unit (i, j, k) that product × relu(c_k) (the product cell passes its products through, ×
  relu(1)), the output layer the table's logits;
- every lesson, schedule and the rest of the datapath are the MLP file's, line for line. The diff replaces the
  `CellMLP` class and its two uses and the `CELL_HIDDEN` constant, and adds a paragraph of docstring.

Its cells hold 116,696 learned numbers, 123,483 in all (the MLP file: 19,488 in its cells, 26,275 in all). The tiny
notch it corresponds to is 2t (the table a dense ReGLU network's outputs; `bases`), the H1 form, as the attention
bridge's MLP file corresponded to its 1e. It was launched before the tiny ladder had finished, on the strength of the
dense ReGLU cell's first runs at (3, 2, 1); the later tiny results (1 of 16 at (3, 2, 2), 0 of 32 for the H1 form) say
it was not a reliable notch.

**Local plumbing check** (`wider_menus/local/local_wide.py`, designed mode, n = 8 per rung): all 14 rungs exact on both
prediction paths with identical logits. The gradient test at the designed point gives every parameter tensor a nonzero
gradient, both cells' included.

**The official run did not certify** (`h1/records/wide_reglu_official.json`; Modal H100, the official manifest untouched,
md5 `a21b3dc9`; the file byte for byte, md5 `17dc352c`; seed 74):

| what | result |
|---|---|
| max certified T, seen / unseen moduli | none: every rung 0/768 on both ladders |
| test, ood_t, ood_n_t | 0/9,999, 0/10,002, 0/10,002 |
| training | the whole 3,600 s budget, 3,122 steps, never frozen; final training loss 2.69 |
| evaluation | 791 s |
| learned numbers | 123,483 |

The schedule ran normally at first: the reading was chosen at step 98, the pieces joined at 250, phase B at 300, the
ladder at 600. But the cells never learned their tables: when the digit facts took over at step 300, their win rate was
0.0000, and the share of the subtraction table exact stayed near 5% to the end, so the division's wiring was never
resolved. On the way the loss blew up, from 8.5 at step 100 to 227 at step 300, before falling back to about 2.8 and
stalling.

**The same file does learn H1, in some runs.** Every run below is the file byte for byte (md5 `17dc352c`):

<!-- H1RUNS -->
| run | device | seed | max T, seen / unseen | every rung 768/768 | test, ood_t, ood_n_t | training s (steps) | evaluation s | md5 |
|---|---|---|---|---|---|---|---|---|
| wide_reglu_cpu_seed1 | CPU | 1 | 64 / 64 | yes | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2531 (20,000) | 641 | 17dc352c |
| wide_reglu_cpu_seed2 | CPU | 2 | none / none | no | 7/9,999, 20/10,002, 5/10,002 | 3600 (2,568) | 9 | 17dc352c |
| wide_reglu_cpu_seed3 | CPU | 3 | none / none | no | 7/9,999, 20/10,002, 5/10,002 | 3601 (2,833) | 8 | 17dc352c |
| wide_reglu_cpu_seed74 | CPU | 74 | none / none | no | 0/9,999, 0/10,002, 0/10,002 | 3601 (2,848) | 1271 | 17dc352c |
| wide_reglu_official | H100 | 74 | none / none | no | 0/9,999, 0/10,002, 0/10,002 | 3601 (3,122) | 791 | 17dc352c |
| wide_reglu_official2 | H100 | 74 | none / none | no | 0/9,999, 0/10,002, 0/10,002 | 3601 (5,248) | 350 | 17dc352c |
<!-- /H1RUNS -->

- **The two H100 runs are the same run.** Their logged losses agree to every printed digit (8.539764 at step 100,
  60.552528 at 200, 227.687012 at 300): the official evaluation of this file is deterministic, so repeating it would
  repeat the failure.
- **Seed 74 fails on a CPU too,** through the same evaluator, so the failure is the seed's trajectory, not the H100's
  arithmetic.
- **Seed 1 certified on a CPU** through the official evaluator (device cpu, seed 1, the manifest otherwise official): T
  = 64 on both ladders, every rung 768/768, test, ood_t and ood_n_t exact. So handwired's ReGLU cell in this shape can
  carry the whole H1 machine. Seeds 2 and 3 failed (both ending with the same 7, 20 and 5 right answers on the three
  splits), so 1 of 4 CPU seeds certified.
- **The CPU diagnostic** (`wider_menus/local/local_wide.py train 700 512 74`, its own batch order) also learned
  everything: the subtraction table at step 300, the product cell at 425, long division by repeated subtraction chosen
  at about 475, frozen at 691, then all 14 depth rungs exact on both prediction paths (128 questions each) with a
  gradient on every parameter.

So the ReGLU cell learns H1 in 1 run of 4, as it learns the tiny machine in some sizes and not others
(§3: 12 of 16 at (3, 2, 1), 1 of 16 at (3, 2, 2)); the attention bridge's ReLU MLP file certified on the H100 and on 8
of 8 CPU seeds. The official H100 manifest fixes seed 74, which fails, so **this file is not certified.** Changing the
file until seed 74 passes was not tried: the tiny evidence never said the cell was reliable, and a fix tuned to one seed
would not say more.

**The 4-copy product circuit does not learn H1** (`h1/submission_wide_x4.py`, md5 `39806d70`: handwired's compiled
product circuit for both cells, every unit in 4 copies, the table learned; designed mode exact on all 14 rungs, every
parameter with a gradient). In the same CPU diagnostic its subtraction table reached only 57-76 of 200 rows and its
product cell 19-23 of 100 by step 700, and every rung was 0, at both cell rates. That matches its tiny result at
(3, 2, 2) (0 of 8, §6), so it got no official run.

**The H1 form at the tiny size** (`bases`: the subtraction table as a dense ReGLU network's outputs on every one-hot
input, read by the compiled products, as in the H1 file): 0 of 32 at bases 3 to 5, both lesson sets. The table learned
directly reaches 8 of 8 at base 4 and 2 of 8 at base 5 (the attention bridge's learned table: 1 of 8 at base 5, where it
stalled on one self-consistent wrong entry, BRIDGE.md §6); the dense ReGLU cell 1 of 24 at bases 4 and 5.

<!-- BASES -->
| notch | lessons | 3,2,1 | 3,2,2 | 4,2,1 | 5,2,1 |
|---|---|---|---|---|---|
| 1. the subtraction table learned (softmax rows) | AC | · | · | 4/4 (87) | 0/4 [0.25] |
| 1. the subtraction table learned (softmax rows) | GC | · | · | 4/4 (137) | 2/4 (112) [0.34] |
| 2d. the subtraction cell a dense ReGLU network | AC | · | · | 1/4 (325) [0.31] | 0/4 [0.21] |
| 2d. the subtraction cell a dense ReGLU network | GC | · | · | 0/4 [0.31] | 0/4 [0.14] |
| 2d. the subtraction cell a dense ReGLU network | ACnb | · | · | 0/4 [0.27] | 0/4 [0.18] |
| 2. the subtraction cell's product circuit, values learned | GC | · | · | 0/4 [0.17] | 0/4 [0.16] |
| 2t. the subtraction table a dense ReGLU network's outputs (the H1 form) | AC | 0/4 [0.45] | 0/4 [0.18] | 0/4 [0.23] | 0/4 [0.05] |
| 2t. the subtraction table a dense ReGLU network's outputs (the H1 form) | GC | 0/4 [0.38] | 0/4 [0.33] | 0/4 [0.24] | 0/4 [0.08] |
<!-- /BASES -->

## 8. Caveats

- **Tiny sizes, few seeds.** Base 3 (4 and 5 in `bases`), 2-digit moduli, 4 seeds per cell; "0/4" means no seed of 4.
- **The program's form was chosen for the comparison.** A handwired programmer would more likely unroll the division
  into one evaluation (a deep feed-forward program); this one is a tick over the attention bridge's grid, run by the
  host loop, so that the two bridges differ only in how a port reads its source. Both forms compute the same cells.
- **The grid is the compile's,** as in the attention bridge: the cell count, the tick count and the answer place come
  from the compiled program; the loosest routing (dense) keeps a causal mask and the cells' kinds.
- **The soft write-back is a choice.** The compiled program's own semantics is "raw"; the product cells make raw soft
  states unstable, so the host normalizes what it writes back after a soft tick (or, for generic cell weights, takes a
  softmax at scale 10). That normalization is outside handwired's affine-plus-ReGLU; the hard path, on which every
  exactness number is measured, snaps instead, as the attention bridge does.
- **Dense routing cannot certify from k = 1.** Its weights are indexed by the source cell's kind, and at one quotient
  digit the shift never reads a select row, so that weight is never trained; at (3, 2, 2) it is. The attention
  bridge's free attention had the same limit.
- **Menus are the attention bridge's,** the 7 designed relative rules plus plain offsets.
- **The multiplexer table mixes two things:** the parametrization of a gate (exp, ReLU, softplus, sigmoid) and the
  normalization. The traces in §4 separate them for one seed only. Softplus's gates use a fixed sharpness (4); others
  were not tried.
- **"Projection" follows the user's words** as relayed by the parent session (a function-preserving lift into a bigger,
  redundant form). The random-projection lift is one choice of lift; Net2Net copies are the other. Snapping onto exact
  programs (projected descent) is reported separately and is not the user's idea.
- **The generic ReGLU cells' reliability depends on how sharply their outputs are read** (softmax of 10 against 3 times
  the values at the tiny size), and the H1 file's read is the MLP file's. The tiny ladder used the sharper read; `over`
  shows the softer one.
- **H1 runs are few:** two H100 runs (identical) and four CPU seeds of one file (1 certified). The CPU seeds change the
  manifest's device and seed only (the attention bridge's protocol); they are not the official run.
- **Run-to-run variation on different machines:** the one pilot of the 4-copy circuit that certified (seed 0, on the
  laptop) did not repeat on Modal with the same seed (0 of 4), and the CPU diagnostic's batch order differs from the
  evaluator's. Results are counted from the Modal records only.

## 9. Reproduce

The JS runs with Bun from the one-layer repo root; it imports handwired's runtime read-only from `~/github/handwired`
(handwired at 257856e, Bun 1.4.3):

```bash
bun cpu_first/handwired_bridge/js/compile.ts
```

Everything else from `cpu_first/` with the sotaku venv's Python. Local runs take seconds to minutes and stay under 1 GB;
the sweeps run on Modal (app and Volume `handwired-bridge`, at most 20 CPU containers):

```bash
~/github/sotaku/venv/bin/python -m handwired_bridge.load
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach handwired_bridge/modal_hb.py --suite ladder
```

```bash
~/github/sotaku/venv/bin/python -m modal run handwired_bridge/modal_hb.py --suite ladder --fetch
```

The other suites go to `--suite` the same way (several at once, comma-separated, share one 20-container cap;
`HB_MAX_CONTAINERS` lowers it): `width` and `width2` (the multiplexers and the menu's width), `diagnose` (the probes),
`bases` (bases 4 and 5, and the H1 form), `over` (copies and softer reads), `lift` (the random-projection lift),
`wstab` and `wlearn` (Net2Net near the compiled point and from random), `updown` (up then down), `proj` (projected
descent). The tables:

```bash
~/github/sotaku/venv/bin/python -m handwired_bridge.fill
```

H1, from `../one-layer-deeper` (app and Volume `handwired-bridge-h1`; records in `handwired_bridge/h1/records/`):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/handwired_bridge/h1/modal_h1.py --file submission_wide_reglu.py --mode official --name wide_reglu_official
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/handwired_bridge/h1/modal_h1.py --file submission_wide_reglu.py --mode plan --plan ../cpu_first/handwired_bridge/h1/plans/wide_reglu_cpu.json
```

```bash
~/github/sotaku/venv/bin/python -m modal run ../cpu_first/handwired_bridge/h1/modal_h1.py --mode fetch
```

The H1 diagnostics (`wider_menus/local/local_wide.py` on Modal CPU containers, app `handwired-bridge-h1diag`; logs in
`handwired_bridge/h1/logs/`):

```bash
~/github/sotaku/venv/bin/python -m modal run ../cpu_first/handwired_bridge/h1/modal_diag_h1.py --plan ../cpu_first/handwired_bridge/h1/plans/diag_cells.json
```

The H1 file's plumbing check, locally (designed mode: every rung exact on both paths; the gradient test):

```bash
~/github/sotaku/venv/bin/python wider_menus/local/local_wide.py $PWD/handwired_bridge/h1/submission_wide_reglu.py designed N=8
```

Files:
- `js/machine.ts`: the machine as one handwired tick (the grid, the cells, the builder code, the host's question I/O);
- `js/compile.ts`: compiles every size, checks it in handwired's evaluators, writes `records/compiled/`;
- `load.py`: handwired's compiled layers in PyTorch, the host loop, the checks (`records/compile.json`);
- `hnet.py`: the compiled weights regrouped (cell circuits, routing, loop) and loosened, the multiplexers;
- `train.py`: the attention bridge's `Run` with this network, the soft write-back, L1, projected descent, compiled
  starts;
- `updown.py`: up then down (blended copies or a population);
- `notches.py`, `suites.py`, `modal_hb.py`: the notches and the sweeps; `summarize.py`, `fill.py`: the tables;
- `probe.py`: the probes (ports, first gradient, the line to the compiled weights, ignored entries, renamed digits);
- `h1/`: the H1 files (`submission_wide_reglu.py`, `submission_wide_x4.py`), their runner (a copy of the attention
  bridge's), the diagnostic runner, plans, records and logs.
