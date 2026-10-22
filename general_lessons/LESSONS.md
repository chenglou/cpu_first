# General lessons: from hand-picked lessons to laws, mined facts and answer-chosen lesson sets

## In plain words

<!-- PLAIN -->
**The question.** The best H1 model ([`submission_wide_anchored.py`](../wider_menus/submission_wide_anchored.py)) is a
small paper-arithmetic calculator that learns its own digit tables and wiring. What teaches it is a set of *lessons*:
statements it can check on each question's own numbers, such as "dividing x by N leaves x, because x is smaller than
N". The litmus test ([LITMUS.md](../litmus/LITMUS.md)) ran it on 26 other tasks and found "a method within its family;
an engineered solution across families": the learning machinery transferred, but the lessons described H1. On a
different formula (y² + 1, 10·y², x cubed, Montgomery's squaring) it built a perfect H1 machine and ignored the answers,
and each of those tasks needed a designer's change. This study makes the lessons general and runs **one file, unchanged,
on every task**.

**What changed.**
- **Laws, stated once.** Every arithmetic law (taking away nothing changes nothing, a − b + b = a, a·(b + 1) = ab + a,
  ...) is now written once, about an operation of the calculator, and applied to whatever numbers a row offers, not
  only to pairs involving N.
- **Facts the model finds itself.** While training, the model keeps the relations between the numbers it reads that hold
  on every row it has seen: which is smaller, which are equal, how many digits, divisibility by 2, 3 and 5. "x is
  smaller than N" is now something it finds, not something it is told (its reader still looks for some order between
  two of the numbers; which two, and which way, it finds). Facts that can hold by accident ("N has at most 7 digits"
  holds on every H1 training row and fails on the test's 8-digit moduli) are reported but never used.
- **Lessons the answers choose.** The lessons that define one step of the division come from one template in two
  versions: long division (bring the number below N) and Montgomery's reduction (clear its lowest digit). Each version
  trains its own complete machine, and only the training answers decide which machine answers. A wrong lesson set
  simply loses.
- **One menu of extra steps for every task**: multiply the square by 10, add a one-digit constant, multiply by the
  round's input. The answers pick.

**What now learns with zero redesign.** Every one of the litmus test's formula failures: y² + 1, 10·y², x^(3^T) and
Montgomery's y²·10⁻⁸ each scored 100% and was exact on every question up to T = 64, on seen and unseen moduli, with the
file byte for byte, on every seed run (Montgomery on 4 of 4 seeds, the other three on 2 of 2 each). So did two tasks
nobody designed for, each on 2 of 2 seeds: 10·y² + 1 (two menu options combined) and y² + 3.

**What still does not, and why.**
- **A formula outside the menu**: y² − 1 (held out) scored 0%. Subtracting 1 is adding N − 1, not a one-digit
  constant, so no combination of the menu computes it; both lesson sets learned their exact machines, no combination
  fit the answers, and the run never settled. The answers choose among what the designer wrote; they cannot invent a
  missing step.
- **The 14 practice datasets with one fixed modulus**: still nothing. One modulus cannot tell the model what its digit
  tokens mean: counted exactly, 8,100 to 453,600 different digit codes agree with every fact in each dataset (with many
  moduli, exactly one does). The information is there, in the answers: in 12 of the 14, exactly one code also
  reproduces the training answers, the true one. But this design takes its digit meanings from facts, and its
  subtraction table gets its meaning from the same facts; finding the code from the answers would mean searching
  thousands to hundreds of thousands of whole codes before the table can be right. General laws do not help here: a
  tiny test shows that laws applied to every pair of numbers still leave the table wrong on one modulus, because algebra
  cannot say which of two digits is larger. Switched on from the start, before the facts, the laws even spoil the
  tables that facts alone get right with many moduli: lessons have an order, facts first.
- **Practice data whose every training row has T = 8** (M4): 0.8%, where the best model certified. The answers that
  choose the lesson set are read only on rows with T ≤ 2 (to keep that choice cheap), and M4 has none, so the choice
  fell to the random start, which picked Montgomery's machine. A gap in the design, fixable in general (read the
  answers at the smallest T present), not fixed here.
- **T's direction on data whose T always has one digit** (M5, and E3 and M4): M5 certified on both seeds, but by luck.
  Nothing in such data tells "16" from "61", so the model keeps the reading direction it starts with: the best model
  started backwards on 2 of its 4 M5 seeds, and this file started right on both M5 seeds but backwards on one E3 seed
  (exact up to T = 8, wrong beyond). A designed prior fixes it (LITMUS.md); no general lesson can.
- **Slower learning.** On one practice dataset the best model learned (E3, seed 74) this file ran out of its 1,500
  steps; with 3,000 it certified. The second lesson set costs time everywhere (next paragraph).

**Does H1 still certify?** Yes, on the untouched official manifest on an H100: every one of 768 questions right at every
T up to 64, on seen and unseen moduli, every split exact, in 1,856 s of training (the best model: 558 s; the second
lesson set's lessons cost time, not correctness). The answers chose long division; the Montgomery set had learned its
own exact machine alongside and lost. On CPU (8-CPU containers, not the official device) all 8 seeds certified as
well, with little to spare: the second lesson set makes the slow phase about twice as slow per step as the best
model's there, so two seeds reached the 3,600 s deadline before freezing and certified only because they were already
exact.

**What this says.** Did general lessons turn "an engineered solution across families" into "a method"? For the formula
failures, yes: the same file now learns every task the litmus test needed a designer for, and two more no one designed
for, because the lessons that described H1 became candidates the answers choose and the extra steps became one shared
menu. But the knowledge did not disappear; it moved. The designer now writes a task-independent space of hypotheses
once (the laws, a grammar of facts, a template with two predicates, a menu of round steps) and the data chooses within
it. That is a method in the ordinary sense of learning, with three limits the results show plainly:
- **the space was written knowing the tasks.** The held-out tasks measure how far it reaches beyond them: compositions
  inside its grammar (10·y² + 1, y² + 3) learn; a formula needing a step it lacks (y² − 1) does not;
- **facts cannot replace information the data lacks.** One modulus does not say what the digits mean, and one-digit T
  does not say which way T runs. The answers say the first, but only to a model that searches whole digit codes, which
  this design does not do;
- **the answers must reach the choice.** The selector reads answers only at T ≤ 2; on M4, where every row has T = 8,
  it never saw one, and the wrong machine answered.
<!-- /PLAIN -->

## Every task at a glance

"The best model unchanged" is LITMUS.md's run of the best model byte for byte on seed 74 (relaxed budget for the
tiers). "The general file, unchanged" is `submission_general.py` byte for byte on every task: H1 through its untouched
official manifest (on an H100, and with the device set to cpu for its 8 CPU seeds), the litmus and held-out tasks
through H1's manifest with their data (at most 1,500 steps, H100s), the tiers through their own manifests with the
relaxed budget (3,600 s, 1,500 steps), on H100s or 8-CPU containers. E3 seed 74's 3,000-step run is left out of this
table (another budget; §4).

<!-- GLANCE -->
| task | the best model unchanged (LITMUS.md) | the smallest change litmus found | the general file, unchanged: runs certified (T = 64 on both ladders) | its score, max T seen / unseen | lesson set and round options chosen |
|---|---|---|---|---|---|
| h1 | certified (WIDE_ANCHORED.md) | none needed | 9/9 | 100.0%, 64 / 64 (×9) | below (0.9986–0.9993), rmul=one radd=0 rcube=none |
| rho | 0.0%, max T 0 / 0 | one switch, +1 before the reduction (17 lines) | 2/2 | 100.0%, 64 / 64 (×2) | below (0.9980–0.9992), rmul=one radd=1 rcube=none |
| ten | 0.0%, max T 0 / 0 | one switch, x10 before the reduction (13 lines) | 2/2 | 100.0%, 64 / 64 (×2) | below (0.9984–0.9992), rmul=ten radd=0 rcube=none |
| cube | 0.1%, max T 0 / 0 | one round option, times the input, and a second multiply operand (21 lines) | 2/2 | 100.0%, 64 / 64 (×2) | below (0.9984–0.9993), rmul=one radd=0 rcube=times_x |
| mont | 0.0%, max T 0 / 0 | its three division lessons rewritten for Montgomery (2/4 seeds), plus a gate check (4/4) | 4/4 | 100.0%, 64 / 64 (×4) | lowzero (0.9970–0.9998), rmul=one radd=0 rcube=none |
| tenrho | - | held out: not a litmus task | 2/2 | 100.0%, 64 / 64 (×2) | below (0.9980–0.9992), rmul=ten radd=1 rcube=none |
| plus3 | - | held out: not a litmus task | 2/2 | 100.0%, 64 / 64 (×2) | below (0.9986–0.9992), rmul=one radd=3 rcube=none |
| minus1 | - | held out: not a litmus task | 0/1 | 0.0%, 0 / 0 | - (no set-and-round combination fits the answers) |
| xfirst | 100.0%, max T 64 / 64 | none needed | 1/1 | 100.0%, 64 / 64 | below (0.9987), rmul=one radd=0 rcube=none |
| m5 | 50.5%, max T 8 / 8 | one tie: T read the way the writer writes (4/4 seeds) | 2/2 | 100.0%, 64 / 64 (×2) | below (0.9991–0.9996), rmul=one radd=0 rcube=none |
| e3 | 100.0%, max T 64 / 64 | none needed (learned unchanged) | 2/5 | 1.6%, 0 / 0; 100.0%, 64 / 64 (×2); 100.0%, 8 / 8 (×2) | - (no set's wiring lessons joined); below (0.9990–0.9999), rmul=one radd=0 rcube=none |
| m4 | 100.0%, max T 64 / 64 | none needed (learned unchanged) | 0/1 | 0.8%, 0 / 0 | - (no set-and-round combination fits the answers) |
| e1 | 1.7%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.0%, 0 / 0 | - (no set's wiring lessons joined) |
| e2 | 0.1%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 1.2%, 0 / 0 | - (no set's wiring lessons joined) |
| e6 | 0.7%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.0%, 0 / 0 | - (no set's wiring lessons joined) |
| e7 | 0.4%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.4%, 0 / 0 | - (no set's wiring lessons joined) |
| e8 | 0.4%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.4%, 0 / 0 | - (no set's wiring lessons joined) |
| e9 | 2.0%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 1.8%, 0 / 0 | - (no set's wiring lessons joined) |
| e10 | 1.7%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 1.3%, 0 / 0 | - (no set's wiring lessons joined) |
| m1 | 0.0%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.0%, 0 / 0 | - (no set's wiring lessons joined) |
| m2 | 0.2%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.1%, 0 / 0 | - (no set's wiring lessons joined) |
| m6 | 2.1%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 1.3%, 0 / 0 | - (no set's wiring lessons joined) |
| m7 | 0.0%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.1%, 0 / 0 | - (no set's wiring lessons joined) |
| m8 | 0.3%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.5%, 0 / 0 | - (no set's wiring lessons joined) |
| m9 | 0.2%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.1%, 0 / 0 | - (no set's wiring lessons joined) |
| m10 | 0.5%, max T 0 / 0 | none found (LITMUS.md §4) | 0/1 | 0.0%, 0 / 0 | - (no set's wiring lessons joined) |
<!-- /GLANCE -->

## Details

### 1. The law library (kind 1)

The best model ([`submission_wide_anchored.py`](../wider_menus/submission_wide_anchored.py), WIDE_ANCHORED.md) teaches
its digit tables and its wiring with lessons it checks on each row's own numbers. Some of them are laws of arithmetic,
some are facts about H1's data, and some define long division's step; in the file they are written together ("x < N, so
x − N borrows", "x mod N = x"). The general file separates them. [`submission_general.py`](submission_general.py) has a
registry, `LAWS`, in which every law is stated once about an operation of the machine and never about a task:

| law | statement | applied to |
|---|---|---|
| sub.identity, sub.self | a − 0 = a and a − a = 0, no borrow | the registers (x and the windows of the machine's own square of x) |
| sub.inverse | (a − b) + b = a, the final borrow equal to the carry | each register minus N again and again; a register against another row's register (new) |
| sub.bijective | every row and column of the subtraction cell uses each digit once | the table |
| sub.order | a < b exactly when a − b borrows, on every top part; subtracting keeps, reverses or wraps the order | every **mined** order fact (below); the entries the batch's subtractions read |
| sub.wrap, sub.zero_minus | (a − b) − b does not borrow for a < b below 10^W; 0 − a borrows when a ≠ 0 | every mined order fact; every mined nonzero fact |
| add.identity, add.commute | 0 + a = a; a + b = b + a (a + b is a − (0 − b)) | N; a register and N; a register and another row's register (new) |
| mul.zero, mul.ladder, mul.distribute | a·0 = 0; a(b + 1) = ab + a; XY = Σ_j X·Y_j·10^j | the product cell; the rows of X·b for X the rows' x and N; X·X under every number rule |
| red.identity, red.zero, red.invariant | red(a·10^s) = a for a < N; red(N·10^s) = 0; red(a·10^s + QN) = a | every mined a < N; N; x with Q another row's x. s is the lesson set's alignment (§3) |
| red.step | a step of the reduction moves its register by multiples of N until the set's predicate holds | §3 |
| csub | the gated subtraction of N keeps a < N and takes a + N to a | x and x + N |

Two things changed besides the bookkeeping:
- **Premises are mined, not assumed.** "x − N borrows" was a law plus the fact x < N. Now sub.order is applied to
  whatever order facts the miner holds (§2). On H1 that is x < N and also T < N; the anchors' top-part lesson is the
  x < N instance.
- **Laws on every pair of numbers (`ALL_PAIRS`).** In the best model every table lesson pairs a number with N (D − N,
  N − x, 0 + N, R + N). LITMUS.md §4 traced the fixed-modulus failure to exactly that: with one N, some digit pairs meet
  in no lesson. The general file also applies sub.inverse and add.commute to a register and another row's register, so
  every digit pair is in some lesson.

**What the all-pairs laws do, tiny first** ([`tiny_table.py`](tiny_table.py): the anchors' table lessons alone, the
reader given, W = 6, from a random start; 3 seeds each; records in `tiny/`):

<!-- TINY_TABLE -->
| moduli | laws applied to | when the algebraic laws join | seeds exact | step exact | entries right at the end (digit, borrow, both) |
|---|---|---|---|---|---|
| 10403 | a register and N | 100 steps after the digit lessons | 0/3 | -, -, - | 54/144/45, 47/156/41, 47/133/44 |
| 10403 | every pair (ALL_PAIRS) | 100 steps after the digit lessons | 0/3 | -, -, - | 34/118/27, 32/154/30, 38/119/26 |
| 10403 | every pair (ALL_PAIRS) | from step 1 | 0/3 | -, -, - | 30/111/27, 35/136/29, 37/111/28 |
| 1517 | a register and N | 100 steps after the digit lessons | 0/3 | -, -, - | 54/122/43, 36/146/31, 45/136/36 |
| 1517 | every pair (ALL_PAIRS) | 100 steps after the digit lessons | 0/3 | -, -, - | 35/111/28, 39/155/38, 40/128/37 |
| 1517 | every pair (ALL_PAIRS) | from step 1 | 0/3 | -, -, - | 32/100/25, 39/119/35, 32/113/22 |
| 323 | a register and N | 100 steps after the digit lessons | 0/3 | -, -, - | 32/133/30, 40/145/34, 44/128/35 |
| 323 | every pair (ALL_PAIRS) | 100 steps after the digit lessons | 0/3 | -, -, - | 47/133/37, 41/153/37, 40/170/40 |
| 323 | every pair (ALL_PAIRS) | from step 1 | 0/3 | -, -, - | 27/98/22, 29/113/24, 38/97/31 |
| 899 | a register and N | 100 steps after the digit lessons | 0/3 | -, -, - | 42/158/37, 34/161/32, 54/153/51 |
| 899 | every pair (ALL_PAIRS) | 100 steps after the digit lessons | 0/3 | -, -, - | 35/154/35, 30/163/30, 32/141/29 |
| 899 | every pair (ALL_PAIRS) | from step 1 | 0/3 | -, -, - | 42/128/33, 41/146/36, 36/145/35 |
| many (4-5 digits) | a register and N | 100 steps after the digit lessons | 3/3 | 77, 73, 71 | 200/200/200, 200/200/200, 200/200/200 |
| many (4-5 digits) | every pair (ALL_PAIRS) | 100 steps after the digit lessons | 3/3 | 77, 73, 71 | 200/200/200, 200/200/200, 200/200/200 |
| many (4-5 digits) | every pair (ALL_PAIRS) | from step 1 | 0/3 | -, -, - | 66/199/66, 34/200/34, 41/200/41 |
<!-- /TINY_TABLE -->

With many moduli, and the laws joining late, the table is exact by step 71–77 on every seed, before the pieces' laws
even join. With one modulus it is not, whether the laws pair numbers with N only or with every other number: the
borrows end 111–170 of 200 right and the digits 30–54. Laws of algebra are true of every relabeling of the digits that keeps the group structure, so
they cannot say which of two digits is larger; only an order fact can, and one modulus orders digits only against N's
own (§2). So the all-pairs laws are kept (they are the general statement, and they cost little), but they do not fix
the fixed-modulus tables.

**Order matters: facts first, laws after.** The rows "from step 1" switch the algebraic laws (with the all-pairs laws)
on from the first step, before the order facts teach anything, instead of 100 steps after the digit lessons. That is
worse, not better. On one modulus no seed became exact and the tables ended worse (borrows 97–146 of 200 right,
against 111–170 when the laws join late). With many moduli, where the late schedule is exact by step 71–77 on every
seed, no seed became exact in 800 steps: the borrows were right by step 50 (the order facts' work), but the digits
stalled at 34–66 of 200. Wrong tables can satisfy the laws too: the laws say how entries relate ((a − b) + b = a,
a + b = b + a, a − 0 = a), not what any digit is. Applied to a random table before the facts have shaped it, they
pull it toward one of those wrong tables, and the facts, arriving later, no longer move it. The general file keeps the best
model's schedule (the pieces' laws join 100 steps after the digit lessons, `LATE_AFTER`), and this is why it must.

### 2. Mined facts (kind 2)

**What the miner does.** Inside training, once the reader ranks its readings (IO stage 3), every step evaluates a
grammar of candidate relations on the batch's fields as read: N, x and T, and in the loss function the answer y. The
grammar (`relations` in the file):
- order: a < b for every ordered pair of fields; equality a = b; "a is the same on every row" (constant);
- nonzero: a ≠ 0; range: a < 10^k for k = 1..8;
- divisibility: a mod d = r and a mod d ≠ 0 for d = 2, 3, 5.

Comparisons are the reader's own: the highest place where two read numbers differ decides, as the reader's ranking
decides x < N. A relation is **in force while it has held on every row of the last 10 batches** (`MINE_WIN`); a batch
where it fails pauses it, and the log counts its failures. Facts are announced in the log when first mined (`LR mined`),
and the facts in force at the end are printed (`MINED`).

**Which mined facts become lessons, and which never do.**
- order facts a < b between fields: sub.order on every top part, sub.wrap, the hard checks of the table and the digit
  gate; and a < N gives red.identity's instances (x and T on H1);
- nonzero facts: sub.zero_minus (0 − a borrows);
- "N mod 2 ≠ 0 and N mod 5 ≠ 0": makes the low-zero reduction admissible (§3: no multiple of N can clear the lowest
  digit otherwise);
- range facts, equality and constants, and every other divisibility fact are reported and never used, because they
  can hold by accident. "N < 10^7" holds on every H1 training row and fails on its unseen 8-digit moduli; "T mod 3 ≠ 0"
  holds on H1's T = 1, 2, 4, 8, 16. A lesson built on either would select machines that only work on the training range.
- Nothing is used before the reading is chosen (IO stage 4), as in the best model.
- One assumption stays designed: the reader learns its digit code by looking for two digit runs that are ordered on
  every row (it finds which two and which way; the larger becomes the modulus). The miner then confirms x < N under
  the code the reader found, alongside every other relation. So "x < N" is found, but inside a reader built to find an
  order.

A fact used as a lesson is a true statement about every row it is applied to (it is applied only while it holds on
every row), so it cannot teach a falsehood about those rows. What an accidental fact could do is select among machines;
that is why only relations between fields are allowed to teach, and divisibility only gates a whole lesson set.

**What the data offers** ([`mine_preview.py`](mine_preview.py) on every dataset's training rows with every field read
right, a diagnostic run after the fact; `preview.json`). The second column counts the digit codes under which x < N
holds on every training row (the reader's own fact; the linear extensions of the digit order it implies), with N read
as written and reversed:

<!-- PREVIEW -->
| task | moduli | digit codes passing x < N (N as written / N reversed) | adjacent digit pairs ordered | facts (every field read right) |
|---|---|---|---|---|
| h1 | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^7, y < N |
| e1 | 1 | 302,400 / 302,400 | 2/9 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 3, T != 0, T < 10^1, T < N, T mod 5 != 0, x != 0, x < 10^3, x < N, y != 0, y < 10^3, y < N |
| e2 | 1 | 40,320 / 362,880 | 2/9 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 4, T != 0, T < 10^1, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^3, x < N, y != 0, y < 10^3, y < N |
| e3 | 27 | 1 / 0 | 9/9 | N != 0, N < 10^4, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 2, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| e4 | 35 | 1 / 0 | 9/9 | N != 0, N < 10^4, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 2, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| e5 | 27 | 1 / 0 | 9/9 | N != 0, N < 10^4, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T mod 5 != 0, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| e6 | 1 | 34,560 / 907,200 | 2/9 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 1, N mod 5 = 2, T != 0, T < 10^1, T < N, T mod 5 != 0, x != 0, x < 10^3, x < N, y != 0, y < 10^3, y < N |
| e7 | 1 | 16,800 / 907,200 | 3/9 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 2, T != 0, T < 10^1, T < N, T mod 5 != 0, x != 0, x < 10^3, x < N, y != 0, y < 10^3, y < N |
| e8 | 1 | 16,800 / 907,200 | 3/9 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 2, T != 0, T < 10^1, T < N, T mod 5 != 0, x != 0, x < 10^3, x < N, y != 0, y < 10^3, y < N |
| e9 | 1 | 120,960 / 907,200 | 2/9 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 4, T != 0, T < 10^1, T < N, T mod 5 != 0, x != 0, x < 10^3, x < N, y != 0, y < 10^3, y < N |
| e10 | 1 | 241,920 / 0 | 2/9 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 1, N mod 5 = 3, T != 0, T < 10^1, T < N, T mod 5 != 0, x != 0, x < 10^3, x < N, y != 0, y < 10^3, y < N |
| m1 | 1 | 90,720 / 1,209,600 | 3/9 | N != 0, N < 10^5, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 3, T != 0, T < 10^2, T < N, T mod 2 = 0, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^5, x < N, y != 0, y < 10^5, y < N |
| m2 | 1 | 16,800 / 0 | 4/9 | N != 0, N < 10^5, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 1, T != 0, T < 10^2, T < N, T mod 2 = 0, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^5, x < N, y != 0, y < 10^5, y < N |
| m3 | 247 | 1 / 0 | 9/9 | N != 0, N < 10^5, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 2, x != 0, x < 10^5, x < N, y != 0, y < 10^5, y < N |
| m4 | 6074 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 3, x != 0, x < 10^7, x < N, y != 0, y < 10^7, y < N |
| m5 | 204 | 1 / 0 | 9/9 | N != 0, N < 10^5, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T mod 2 = 0, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^5, x < N, y != 0, y < 10^5, y < N |
| m6 | 1 | 81,000 / 1,209,600 | 3/9 | N != 0, N < 10^4, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 2, T != 0, T < 10^1, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| m7 | 1 | 8,100 / 1,209,600 | 4/9 | N != 0, N < 10^4, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 3, T != 0, T < 10^1, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| m8 | 1 | 453,600 / 1,209,600 | 2/9 | N != 0, N < 10^4, N constant, N mod 2 = 1, N mod 3 = 1, N mod 5 = 3, T != 0, T < 10^1, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| m9 | 1 | 10,800 / 1,209,600 | 4/9 | N != 0, N < 10^4, N constant, N mod 2 = 1, N mod 3 = 1, N mod 5 = 2, T != 0, T < 10^1, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| m10 | 1 | 10,710 / 1,209,600 | 4/9 | N != 0, N < 10^4, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 4, T != 0, T < 10^1, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| xfirst | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^7, y < N |
| mont | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^6, y < N |
| ten | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^7, y < N |
| rho | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^7, y < N |
| cube | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^7, y < N |
| xy | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^7, y < N |
| tenrho | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y < 10^6, y < N |
| plus3 | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y < 10^7, y < N |
| minus1 | 430 | 1 / 0 | 9/9 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y < 10^7, y < N |
<!-- /PREVIEW -->

- **H1 and every sampled-modulus dataset:** exactly one code passes, and reading N backwards passes none: the facts pin
  every digit's meaning and N's direction. The miner finds x < N, T < N, y < N, the nonzero facts and N's being odd
  and not a multiple of 3 or 5, plus the accidental range and T-divisibility facts.
- **The 14 fixed-modulus datasets:** 8,100 (M7) to 453,600 (M8) codes pass, and in 12 of the 14 as many again with N
  read backwards (E10 and M2 rule the backwards reading out). The miner's extra facts there are "N is constant" and N's
  own residues (E1: N mod 3 = 2, N mod 5 = 3); none of them orders two digits that are not N's.

**What the model mined in training** (its own log, the facts in force at the end of each run):

<!-- MINED -->
| run | task | facts in force at the end (the model's own miner) |
|---|---|---|
| gen_cube_s1 | cube | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_cube_s74 | cube | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_e10_relaxed | e10 | N < 10^3, T < 10^1, x < 10^3, y < 10^3 |
| gen_e1_reader_frozen | e1 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 3, T != 0, T < 10^1, T < N, T mod 5 != 0, x != 0, x < 10^3, x < N, y != 0, y < 10^3, y < N |
| gen_e1_relaxed | e1 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 4, T < 10^1, T < N, x != 0, x < 10^1, x < N, x constant, x mod 2 = 1, x mod 3 = 0, x mod 5 = 4, y < 10^3, y < N |
| gen_e2_relaxed | e2 | N != 0, N < 10^3, N constant, N mod 2 = 0, N mod 3 = 2, N mod 5 = 3, T != 0, T < 10^2, T < N, T constant, T mod 2 = 1, T mod 3 = 0, T mod 5 = 4, x < 10^2, x < N, x < T, y < 10^3, y < N |
| gen_e3_relaxed | e3 | N != 0, N < 10^4, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 2, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| gen_e3_relaxed_s1 | e3 | N != 0, N < 10^4, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 2, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| gen_e3_relaxed_s1_h100 | e3 | N != 0, N < 10^4, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 2, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| gen_e3_relaxed_s2 | e3 | N != 0, N < 10^4, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 2, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| gen_e3_relaxed_s2_h100 | e3 | N != 0, N < 10^4, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 2, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| gen_e3_steps3000_h100 | e3 | N != 0, N < 10^4, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 2, x != 0, x < 10^4, x < N, y != 0, y < 10^4, y < N |
| gen_e6_relaxed | e6 | N < 10^1, N < x, N = T, N constant, N mod 2 = 0, N mod 3 = 0, N mod 5 = 0, T < 10^1, T < x, T constant, T mod 2 = 0, T mod 3 = 0, T mod 5 = 0, x != 0, x < 10^2, x constant, x mod 2 = 0, x mod 3 = 0, x mod 5 = 3, y < 10^3 |
| gen_e7_relaxed | e7 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 0, N mod 5 = 2, T < 10^1, T < N, T < x, T constant, T mod 2 = 0, T mod 3 = 0, T mod 5 = 0, x != 0, x < 10^2, x < N, x constant, x mod 2 = 0, x mod 3 = 0, x mod 5 = 3, y < 10^3, y < N |
| gen_e8_relaxed | e8 | N != 0, N < 10^3, N constant, N mod 2 = 1, N mod 3 = 0, N mod 5 = 2, T < 10^1, T < N, T < x, T constant, T mod 2 = 0, T mod 3 = 0, T mod 5 = 0, x != 0, x < 10^2, x < N, x constant, x mod 2 = 0, x mod 3 = 0, x mod 5 = 3, y < 10^3, y < N |
| gen_e9_relaxed | e9 | N != 0, N < 10^3, N constant, N mod 2 = 0, N mod 3 = 2, N mod 5 = 3, T != 0, T < 10^2, T < N, T constant, T mod 2 = 1, T mod 3 = 0, T mod 5 = 4, x < 10^2, x < N, x < T, y < 10^3, y < N |
| gen_h1_cpu_seed1 | h1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_h1_cpu_seed2 | h1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_h1_cpu_seed3 | h1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_h1_cpu_seed4 | h1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_h1_cpu_seed5 | h1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_h1_cpu_seed6 | h1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_h1_cpu_seed7 | h1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_h1_cpu_seed74 | h1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^6, y < N |
| gen_h1_official | h1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_m10_relaxed | m10 | N < 10^1, N < T, N < y, N constant, N mod 2 = 0, N mod 3 = 0, N mod 5 = 0, T != 0, T < 10^1, T mod 5 != 0, x < 10^4, y != 0, y < 10^4 |
| gen_m1_reader_frozen | m1 | N != 0, N < 10^5, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 3, T != 0, T < 10^2, T < N, T mod 2 = 0, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^5, x < N, y != 0, y < 10^5, y < N |
| gen_m1_relaxed | m1 | N != 0, N < 10^5, N constant, N mod 2 = 0, N mod 3 = 1, N mod 5 = 2, T != 0, T < 10^2, T < N, T mod 3 != 0, x < 10^5, y != 0, y < 10^5, y < N |
| gen_m2_relaxed | m2 | N != 0, N < 10^5, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 1, T != 0, T < 10^2, T < N, x != 0, x < 10^5, x < N, y != 0, y < 10^5, y < N |
| gen_m4_relaxed_h100 | m4 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T constant, T mod 2 = 0, T mod 3 = 2, T mod 5 = 3, x != 0, x < 10^7, x < N, y != 0, y < 10^7, y < N |
| gen_m5_relaxed_s1 | m5 | N != 0, N < 10^5, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T mod 2 = 0, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^5, x < N, y != 0, y < 10^5, y < N |
| gen_m5_relaxed_s74 | m5 | N != 0, N < 10^5, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^1, T < N, T mod 2 = 0, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^5, x < N, y != 0, y < 10^5, y < N |
| gen_m6_relaxed | m6 | N != 0, N < 10^4, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 4, T < 10^2, T < N, x != 0, x < 10^2, x < N, x constant, x mod 2 = 0, x mod 3 = 2, x mod 5 = 3, y != 0, y < 10^4 |
| gen_m7_relaxed | m7 | N != 0, N < 10^4, N constant, N mod 2 = 1, N mod 3 = 2, N mod 5 = 4, T != 0, T < 10^1, T < N, T mod 3 != 0, T mod 5 != 0, x < 10^1, x < N, x < T, x < y, x constant, x mod 2 = 0, x mod 3 = 0, x mod 5 = 0, y != 0, y < 10^4 |
| gen_m8_relaxed | m8 | N != 0, N < 10^4, N constant, N mod 2 = 1, N mod 3 = 1, N mod 5 = 1, T < 10^1, T < N, x < 10^4, x < N, y < 10^4, y < N |
| gen_m9_relaxed | m9 | N < 10^1, N constant, N mod 2 = 0, N mod 3 = 0, N mod 5 = 0, T < 10^1, T mod 2 = 0, x < 10^4, y < 10^4 |
| gen_minus1_s74 | minus1 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y < 10^6, y < N |
| gen_mont_s1 | mont | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_mont_s2 | mont | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_mont_s3 | mont | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_mont_s74 | mont | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_plus3_s1 | plus3 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T < y, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_plus3_s74 | plus3 | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T < y, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_rho_s1 | rho | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T < y, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_rho_s74 | rho | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^6, y < N |
| gen_ten_s1 | ten | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_ten_s74 | ten | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_tenrho_s1 | tenrho | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
| gen_tenrho_s74 | tenrho | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^7, x < N, y != 0, y < 10^6, y < N |
| gen_xfirst_s74 | xfirst | N != 0, N < 10^7, N mod 2 = 1, N mod 3 != 0, N mod 5 != 0, T != 0, T < 10^2, T < N, T mod 3 != 0, T mod 5 != 0, x != 0, x < 10^6, x < N, y != 0, y < 10^6, y < N |
<!-- /MINED -->

- **On H1's prompts** (H1 and every litmus and held-out task): on seed 74, x < N, T < N and the nonzero facts were mined
  at step 174, the step the reader chose its reading, and y < N three steps later; on seeds 1, 2, 3 and 5, at steps
  113–135, between the reader's ranking and its choice. The digit lessons followed (seed 74: step 216–219), as in the
  best model they follow the reading. The same runs also show the miner catching **facts true by accident**: on seed 74
  at step 165–166, under a reading that was still wrong, "x = T" and "x < 10^2" held on ten batches in a row (x and T
  were being read as the same short number); they lapsed as soon as the reading was right, and as equality and range
  facts they would never have taught anything. Later, "T < x" and "T < y" were mined (seed 74: steps 413 and 389; the
  other seeds: 118–162) because x and y happened to be at least 17 on ten batches in a row (training T is at most 16);
  T < x, an order fact between fields, teaches the table on the batches where it holds, which is harmless (it is true on
  every row it is applied to), and lapses when a batch breaks it.
- **On a fixed modulus**: the reader chose no reading on ten of the fourteen tiers and a scrambled one on the other four
  (§4), so the mined facts were either never used or about misread numbers. The facts in force at the end describe
  misread numbers (E2: "T constant", "x < T", even "N mod 2 = 0"; E6: "N = T", "N < 10^1"). Where N looked even, the
  low-zero lesson set was not admitted, as the admissibility rule says.

**Rule 14.** The miner computes over the rows the evaluator hands to the model during training, inside the model's
forward pass and loss; it never reads the data files, and it makes up no numbers. That is the model learning from its
batches, not participant data inspection. It is still a gray area: a strict reader could call a model that searches its
own batches for regularities "data inspection", and the grammar (which relations to look for) is designed. The miner's
divisibility verdicts also use the reader's digit labels as numbers (the lowest digit's parity, the digit sum mod 3),
which is bookkeeping the model does in Python, not a learned operation; they only decide whether the low-zero lesson set
may run.

### 3. Answer-chosen lessons (kind 3, and the lever)

**The problem they solve.** In the best model the answers only choose among a few reading switches at the end; the
lessons decide the machine. On `mont`, `rho`, `ten` and `cube` the lessons describe long division of x², the file
built exactly that machine, and the answers could not override it (LITMUS.md §4). Two things were missing: a way for a
wrong lesson to lose, and machines for the answers to choose.

**Lesson sets from one template.** The division's step-level lessons (the division ladder and the step law) and its
whole-reduction laws (red.identity, red.zero, red.invariant) are now generated by one template from a predicate of a
small grammar, `PREDICATES`:
- **below**: the chain moves its register *down* by multiples of N until it is below N. Ladder: from x + Nq (q = 0..9)
  one chain gives x. Step law: the register split into its upper places and its lowest digit; bring the digit back in,
  run the chain: x. Alignment s = 0: red(a) = a for every mined a < N, red(N) = 0, red(x + QN) = x. These are the best
  model's lessons, unchanged.
- **lowzero**: the chain moves its register *up* by multiples of N until its lowest digit is 0. Ladder: from 10x + Nq
  one chain gives 10x (q = 0) or 10(x + N). Step law: the register with part of its lowest digit left in place and the
  rest coming in, added where it lands (writing it over the place would lose the part left; the split is made with the
  learned table); the chain and the post move give x or x + N. Alignment s = W: red(a·10^W) = a, red(N·10^W) = 0,
  red(x·10^W + QN) = x. Admissible only when the miner holds N mod 2 ≠ 0 and N mod 5 ≠ 0.

Both sets also carry csub, the law of the gated subtraction (from x + N it subtracts N, from x it holds), which pins the
commit gate's meaning; the gate is shared with the countdown, so a set whose chain wants the gate the other way round
cannot break the loop over T (LITMUS.md §4: Montgomery's trial-multiples form did exactly that).

These two instances are the ones H1 and Montgomery's squaring need. The template was written knowing both; what is new
is that neither is chosen by the designer for a task: the same file holds both sets on every task, and the answers pick.

**Each set has its own wiring.** Every lesson set owns a full copy of the division's switches (`sws`), taught only by
its own lessons: the set's ladder for the chain block and the gate, then its step law, then its whole-reduction laws and
the answers for the reading block. Tables, multiply layout, reader and gate are shared. So on every task the model
trains two complete reductions side by side: on H1's prompts the lowzero set learns a Montgomery reduction and the below
set long division, whichever task the answers come from.

**The selector, trained by the answers alone.** A logit per lesson set (`lsel`) and the shared round menu's switches
(`rnd`) are trained by one factorized mixture: for every set whose I/O lessons have joined and every combination of the
round's options, "this set's argmax machine with these options reproduces the rows with T ≤ 2". No lesson loss reaches
the selector. This is the lever:
- a wrong lesson set loses: its machine reproduces no answer, so its weight only falls;
- **"switch every lesson off" cannot win**: a set without lessons learns no machine, so it never fits an answer, and
  its lesson loss (zero) is not part of the selector's objective;
- a lesson set cannot force its machine: the prediction uses the selected set's switches (straight-through over the
  sets, so the final logits' gradient reaches the selector and the selected set's switches; §5's gradient test).

All three need answers to reach the selector, and it reads only the rows with T ≤ 2 (`REAL_MAX_T`, the best model's
rule for its reading lesson). On M4, whose training rows all have T = 8, none did, and the selector stayed at its random
start (§4).

**A shared round menu.** The litmus variants each added one option. The general file has all of them, the same on every
task: the square times 10 or not (`rmul`), plus a one-digit constant 0–9 (`radd`), and after the reduction times the
round's own input and reduced again or not (`rcube`): 40 combinations. Only the answers teach it, through the
selector's mixture.

**Tiny first** ([`tiny_general.py`](tiny_general.py): wide.py's value machine with every table at the truth, W = 3,
2-digit training moduli, certification at T = 1, 2, 4, 8 on 1- to 3-digit moduli; division menu level 3: 3,600 chain
combinations, 210 step, 144 reading; 34 runs on Modal CPUs, records in `tiny/`):

<!-- TINY_GEN -->
| task | candidates | selector trained by | answered rows (share) | seeds certified (T up to 8, every width) | set chosen (its weight) | round options chosen | each set's chain, by seed |
|---|---|---|---|---|---|---|---|
| cube | below,lowzero | answers only | 0.02 | 3/3 | below (0.949), below (0.990), below (0.994) | rmul=1 radd=0 rcube=times_x | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD; s2: below D-M sub borrow, lowzero Rp-Mq add zD |
| mod | below,lowzero | answers only | 0.002 | 1/1 | below (0.997) | rmul=1 radd=0 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD |
| mod | below,lowzero | lesson loss (soft gates) | 0.02 | 2/2 | below (0.990), below (0.996) | rmul=1 radd=0 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD |
| mod | below,lowzero,none | answers only | 0.02 | 2/2 | below (0.987), below (0.994) | rmul=1 radd=0 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD |
| mod | below,lowzero | answers only | 0.02 | 3/3 | below (0.945), below (0.990), below (0.995) | rmul=1 radd=0 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD; s2: below D-M sub borrow, lowzero Rp-Mq add zD |
| mont | below,lowzero,none | answers only | 0.02 | 2/2 | lowzero (0.984), lowzero (0.991) | rmul=1 radd=0 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD |
| mont | below,lowzero | answers only | 0.02 | 3/3 | lowzero (0.972), lowzero (0.986), lowzero (0.995) | rmul=1 radd=0 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD; s2: below D-M sub borrow, lowzero Rp-Mq add zD |
| montrho | below,lowzero | answers only | 0.02 | 3/3 | lowzero (0.970), lowzero (0.987), lowzero (0.993) | rmul=1 radd=1 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD; s2: below D-M sub borrow, lowzero Rp-Mq add zD |
| plus2 | below,lowzero | answers only | 0.002 | 1/1 | below (0.997) | rmul=1 radd=2 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD |
| plus2 | below,lowzero | answers only | 0.02 | 3/3 | below (0.949), below (0.992), below (0.994) | rmul=1 radd=2 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD; s2: below D-M sub borrow, lowzero Rp-Mq add zD |
| rho | below,lowzero | answers only | 0.002 | 1/1 | below (0.996) | rmul=1 radd=1 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD |
| rho | below,lowzero | answers only | 0.02 | 3/3 | below (0.958), below (0.992), below (0.994) | rmul=1 radd=1 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD; s2: below D-M sub borrow, lowzero Rp-Mq add zD |
| ten | below,lowzero | answers only | 0.002 | 1/1 | below (0.996) | rmul=10 radd=0 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD |
| ten | below,lowzero | answers only | 0.02 | 3/3 | below (0.938), below (0.989), below (0.995) | rmul=10 radd=0 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD; s2: below D-M sub borrow, lowzero Rp-Mq add zD |
| tenrho | below,lowzero | answers only | 0.02 | 3/3 | below (0.952), below (0.989), below (0.995) | rmul=10 radd=1 rcube=none | s0: below Rp-Mq sub borrow, lowzero M-D add zD; s1: below Rp-Mq sub borrow, lowzero M-D add zD; s2: below D-M sub borrow, lowzero Rp-Mq add zD |
<!-- /TINY_GEN -->

- Every task certified on every seed with the right set and round options: long division and the plain square for mod,
  Montgomery for mont, +1, ×10 and times-the-input for rho, ten and cube, and the held-out compositions (10y² + 1,
  y² + 2, Montgomery of y² + 1) that no design decision looked at.
- Both sets learn a correct reduction on every task, in either of two forms each (below: trial multiples or repeated
  subtraction; lowzero: repeated addition or trial multiples testing the register's digit). The answers then choose.
- **Trap: a set without lessons.** A third candidate with no lessons ("none") ended at 0.3–0.4% weight; the right set
  still won.
- **Trap: few answers.** With about 8 answered rows in the whole 4,000-row pool (label share 0.002), mod, rho, ten and
  plus2 still chose right and certified (those runs went to the 1,500-step cap only because the tiny harness stops
  after 30 batches in a row that each hold an answered row, and only about one batch in four holds one).
- **Trap: lesson gates trained by the lesson loss.** The alternative formulation, a soft gate per lesson set multiplying
  its lesson loss and trained by that same loss, has its optimum at every gate 0. In the tiny core the gates fell to
  0.06 and 0.005 but only after the lessons had converged, and Adam rescales a shrinking gradient, so it certified by
  timing, not by design. The selector formulation has no such degenerate optimum.
- **Impostor lesson sets** did not appear: in every run the selector ended on the right set and round options, and the
  chosen machine certified on moduli and depths it never trained on.
- **Trap: facts true by accident.** The tiny miner held "N < 10^2", "x < 10^2" and "y < 10^2" on every training batch
  (the training moduli have 2 digits), all false on the 3-digit certification moduli. Range facts never teach, and
  every run certified on the 3-digit moduli too. §2 shows the same at full scale ("N < 10^7" on H1, and transient facts
  under a half-trained reading).

### 4. The litmus failures, re-run with zero redesign

Every run below is `submission_general.py` byte for byte (md5 `ddfdc065…`). The litmus and held-out tasks have H1's
prompts row for row (litmus/tasks.py's generators, three answers added in [`tasks_general.py`](tasks_general.py)), so
every label-free lesson sees the same numbers as on H1 and the runs follow H1's schedule step for step until the answers
first matter (§3 of LITMUS.md).

<!-- RUNS -->
| run | task | file | device | score | splits | max T seen / unseen | steps | training s | evaluation s | where it stopped | lesson set chosen (selector) | round options | each set's chain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen_cube_s1 | cube | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1269 | 101 | froze at step 630 | below (lowzero:0.0007,below:0.9993) | rmul=one radd=0 rcube=times_x | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_cube_s74 | cube | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1087 | 88 | froze at step 646 | below (lowzero:0.0016,below:0.9984) | rmul=one radd=0 rcube=times_x | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_e10_relaxed | e10 | the file | CPU | 1.3% | ood 1.6%, test 1.0% | 0 / 0 | 1500 | 2049 | 837 | subtraction table never exact (12/200 entries right) | - (no set's wiring lessons joined) | - | - |
| gen_e1_reader_frozen | e1 | reader_frozen | CPU | 4.7% | ood 6.0%, test 3.3% | 0 / 0 | 1500 | 2907 | 5 | subtraction table never exact (46/200 entries right) | - (no set's wiring lessons joined) | - | - |
| gen_e1_relaxed | e1 | the file | CPU | 0.0% | ood 0.0%, test 0.0% | 0 / 0 | 1500 | 595 | 799 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_e2_relaxed | e2 | the file | CPU | 1.2% | ood 0.7%, test 1.7% | 0 / 0 | 1500 | 434 | 510 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_e3_relaxed | e3 | the file | CPU | 1.6% | ood 2.6%, test 0.5% | 0 / 0 | 1500 | 1234 | 4 | subtraction table never exact (110/200 entries right) | - (no set's wiring lessons joined) | - | - |
| gen_e3_relaxed_s1 | e3 | the file | CPU | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 685 | 3613 | 167 | certified without freezing | below (lowzero:0.0007,below:0.9993) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_e3_relaxed_s1_h100 | e3 | the file | H100 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1483 | 3602 | 41 | certified without freezing | below (lowzero:0.0001,below:0.9999) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_e3_relaxed_s2 | e3 | the file | CPU | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 865 | 3608 | 166 | max T 8 | below (lowzero:0.0010,below:0.9990) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: trial multiples |
| gen_e3_relaxed_s2_h100 | e3 | the file | H100 | 100.0% | ood 100.0%, test 100.0% | 8 / 8 | 1500 | 2823 | 39 | max T 8 | below (lowzero:0.0003,below:0.9997) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: trial multiples |
| gen_e3_steps3000_h100 | e3 | the file | H200 | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 2383 | 3602 | 32 | certified without freezing | below (lowzero:0.0001,below:0.9999) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_e6_relaxed | e6 | the file | CPU | 0.0% | ood 0.0%, test 0.0% | 0 / 0 | 1500 | 701 | 755 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_e7_relaxed | e7 | the file | CPU | 0.4% | ood 0.0%, test 0.8% | 0 / 0 | 1500 | 630 | 742 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_e8_relaxed | e8 | the file | CPU | 0.4% | ood 0.0%, test 0.8% | 0 / 0 | 1500 | 555 | 640 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_e9_relaxed | e9 | the file | CPU | 1.8% | ood 2.2%, test 1.3% | 0 / 0 | 1500 | 674 | 468 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_h1_cpu_seed1 | h1 | the file | CPU | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 2857 | 525 | froze at step 629 | below (lowzero:0.0009,below:0.9991) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_h1_cpu_seed2 | h1 | the file | CPU | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 644 | 3604 | 587 | froze at step 644 | below (lowzero:0.0007,below:0.9993) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: trial multiples |
| gen_h1_cpu_seed3 | h1 | the file | CPU | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 586 | 3613 | 310 | certified without freezing | below (lowzero:0.0012,below:0.9988) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_h1_cpu_seed4 | h1 | the file | CPU | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 3583 | 202 | froze at step 750 | below (lowzero:0.0008,below:0.9992) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: trial multiples |
| gen_h1_cpu_seed5 | h1 | the file | CPU | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 3550 | 542 | froze at step 646 | below (lowzero:0.0007,below:0.9993) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_h1_cpu_seed6 | h1 | the file | CPU | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 3269 | 198 | froze at step 646 | below (lowzero:0.0010,below:0.9990) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_h1_cpu_seed7 | h1 | the file | CPU | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 11853 | 3600 | 507 | froze at step 717 | below (lowzero:0.0013,below:0.9987) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_h1_cpu_seed74 | h1 | the file | CPU | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 640 | 3608 | 624 | certified without freezing | below (lowzero:0.0014,below:0.9986) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_h1_official | h1 | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 20000 | 1856 | 97 | froze at step 645 | below (lowzero:0.0014,below:0.9986) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_m10_relaxed | m10 | the file | CPU | 0.0% | ood 0.0%, test 0.0% | 0 / 0 | 1500 | 430 | 20 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_m1_reader_frozen | m1 | reader_frozen | CPU | 2.3% | ood 2.6%, test 1.9% | 0 / 0 | 1500 | 2198 | 6 | subtraction table never exact (39/200 entries right) | - (no set's wiring lessons joined) | - | - |
| gen_m1_relaxed | m1 | the file | CPU | 0.0% | ood 0.0%, test 0.1% | 0 / 0 | 1500 | 1992 | 17 | subtraction table never exact (56/200 entries right) | - (no set's wiring lessons joined) | - | - |
| gen_m2_relaxed | m2 | the file | CPU | 0.1% | ood 0.1%, test 0.1% | 0 / 0 | 1500 | 1948 | 7 | subtraction table never exact (36/200 entries right) | - (no set's wiring lessons joined) | - | - |
| gen_m4_relaxed_h100 | m4 | the file | H100 | 0.8% | ood 1.0%, test 0.6% | 0 / 0 | 1500 | 2648 | 70 | stopped in phase C: hard checks 1.0000, no set-and-round fits (no row has T <= 2) | lowzero (lowzero:0.5753,below:0.4247) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_m5_relaxed_s1 | m5 | the file | CPU | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 827 | 3613 | 235 | certified without freezing | below (lowzero:0.0004,below:0.9996) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_m5_relaxed_s74 | m5 | the file | CPU | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 821 | 3611 | 110 | certified without freezing | below (lowzero:0.0009,below:0.9991) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_m6_relaxed | m6 | the file | CPU | 1.3% | ood 1.4%, test 1.3% | 0 / 0 | 1500 | 448 | 951 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_m7_relaxed | m7 | the file | CPU | 0.1% | ood 0.0%, test 0.2% | 0 / 0 | 1500 | 386 | 294 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_m8_relaxed | m8 | the file | CPU | 0.5% | ood 0.7%, test 0.3% | 0 / 0 | 1500 | 2732 | 10 | subtraction table never exact (47/200 entries right) | - (no set's wiring lessons joined) | - | - |
| gen_m9_relaxed | m9 | the file | CPU | 0.1% | ood 0.0%, test 0.2% | 0 / 0 | 1500 | 320 | 252 | reading never chosen (io stage 3) | - (no set's wiring lessons joined) | - | - |
| gen_minus1_s74 | minus1 | the file | H100 | 0.0% | ood_n_t 0.1%, ood_t 0.0%, test 0.0% | 0 / 0 | 1013 | 3603 | 163 | stopped in phase C: hard checks 1.0000, no set-and-round fits (answers 0) | below (lowzero:0.2474,below:0.7526) | rmul=one radd=4 rcube=times_x | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_mont_s1 | mont | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1171 | 85 | froze at step 633 | lowzero (lowzero:0.9992,below:0.0008) | rmul=one radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_mont_s2 | mont | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1823 | 94 | froze at step 652 | lowzero (lowzero:0.9994,below:0.0006) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: trial multiples |
| gen_mont_s3 | mont | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1509 | 109 | froze at step 659 | lowzero (lowzero:0.9970,below:0.0030) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_mont_s74 | mont | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1068 | 70 | froze at step 698 | lowzero (lowzero:0.9998,below:0.0002) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_plus3_s1 | plus3 | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1281 | 53 | froze at step 630 | below (lowzero:0.0008,below:0.9992) | rmul=one radd=3 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_plus3_s74 | plus3 | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 879 | 41 | froze at step 646 | below (lowzero:0.0014,below:0.9986) | rmul=one radd=3 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_rho_s1 | rho | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1022 | 44 | froze at step 630 | below (lowzero:0.0008,below:0.9992) | rmul=one radd=1 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_rho_s74 | rho | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1012 | 46 | froze at step 641 | below (lowzero:0.0020,below:0.9980) | rmul=one radd=1 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_ten_s1 | ten | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1325 | 60 | froze at step 630 | below (lowzero:0.0008,below:0.9992) | rmul=ten radd=0 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_ten_s74 | ten | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1406 | 84 | froze at step 646 | below (lowzero:0.0016,below:0.9984) | rmul=ten radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_tenrho_s1 | tenrho | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1024 | 45 | froze at step 630 | below (lowzero:0.0008,below:0.9992) | rmul=ten radd=1 rcube=none | lowzero: montgomery, trial multiples (the register's digit tested); below: repeated subtraction |
| gen_tenrho_s74 | tenrho | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1454 | 90 | froze at step 641 | below (lowzero:0.0020,below:0.9980) | rmul=ten radd=1 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
| gen_xfirst_s74 | xfirst | the file | H100 | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 853 | 39 | froze at step 562 | below (lowzero:0.0013,below:0.9987) | rmul=one radd=0 rcube=none | lowzero: montgomery, repeated addition; below: repeated subtraction |
<!-- /RUNS -->

**The other formulas: learned, with no change.** The runs repeat H1's schedule step for step until the answers matter.
On seed 74: the reading chosen at step 174, the digit lessons at 219, phase C at 404–410, the below set's wiring lessons
at 426–432 and the lowzero set's at 465–472; on seed 1, the same on every task, earlier: 132, 178, 398, 420 and 428.
From then on the selector sees both sets' machines on the answered rows:
- `mont`: only the lowzero set's machine (on seed 74 a Montgomery reduction by repeated addition: start empty, digits
  brought in upwards and added at place 0, add N while the lowest digit is not 0, move down one place, add the top
  half, one gated subtraction) reproduced the answers; the selector ended at 0.9998 on it, and the run froze at step
  698. Seeds 1–3 certified too (selector 0.9970–0.9994, frozen at 633–659; on seed 1 the lowzero machine was the
  trial-multiples form that tests the register's lowest digit). LITMUS.md needed three rewritten division lessons and a
  fourth gate check for this task;
- `rho`, `cube` and the held-out `plus3`: the below set's long division (repeated subtraction) with the round options
  +1, × the input and +3 reproduced them; selector 0.998–0.9993, frozen at steps 641–646 on seed 74 and 630 on seed 1.
  LITMUS.md needed one new switch or round option each for rho and cube;
- `ten` and the held-out `tenrho` (10y² + 1, a combination of two menu options nothing was designed for): the below
  set with ×10, and ×10 with +1; frozen at steps 646 and 641 on seed 74, both at 630 on seed 1;
- `xfirst` (H1 with the prompt's fields reordered, which the best model already learned unchanged): certified again;
- the choice is quick: by step 500, 30–80 steps after both sets' machines could compute, the selector stood at
  0.985–0.999 on the right set on every task and seed, with the right round options; the runs then froze 200 exact
  steps later;
- the losing set fit none of the answered rows on any task but one: on one logged batch of `ten`, Montgomery's machine
  under some round option matched 6% of the answered rows by coincidence, never enough to compete (selector 0.998).
- on every task both sets learned a correct reduction (the same two wirings on every task of one seed, since the
  prompts are H1's: on seed 74 Montgomery by repeated addition and long division by repeated subtraction; on seed 1
  Montgomery by trial multiples testing the register's digit, and repeated subtraction): the answers chose between two
  working machines, not between a working one and noise.

**The 14 fixed-modulus tiers: still nothing.** All 14 failed (best score 1.8%). Ten never chose a reading (E1, E2,
E6–E9, M6, M7, M9, M10: the reader's posterior stayed below 0.99, so the table never got a lesson). Four chose one (E10,
M1, M2, M8) with a scrambled digit code (0–4 of the 10 tokens right), and their tables never became exact (12–56 of 200
entries). LITMUS.md saw the same two failures in another split (7 and 7): whether the reader commits on one modulus is a
coin flip of its random start, and the general file's extra parameters change that start. Either way, nothing it can
learn from one modulus fixes the meaning of the digits:

<!-- CODES -->
| task | N | codes passing x < N (N as written) | N reversed also passes? | answered rows the file uses (T <= 2) | codes that also reproduce every one of those answers | the true code among them |
|---|---|---|---|---|---|---|
| e1 | 323 | 302,400 | yes, 302,400 codes | 400 | 1 | yes |
| e2 | 899 | 40,320 | yes, 362,880 codes | 1,280 | 1 | yes |
| e6 | 247 | 34,560 | yes, 907,200 codes | 294 | 1 | yes |
| e7 | 287 | 16,800 | yes, 907,200 codes | 352 | 1 | yes |
| e8 | 287 | 16,800 | yes, 907,200 codes | 352 | 1 | yes |
| e9 | 299 | 120,960 | yes, 907,200 codes | 400 | 1 | yes |
| e10 | 403 | 241,920 | no | 528 | 1 | yes |
| m1 | 10403 | 90,720 | yes, 1,209,600 codes | 0 | - (no answered rows) | - |
| m2 | 38021 | 16,800 | no | 0 | - (no answered rows) | - |
| m6 | 1517 | 81,000 | yes, 1,209,600 codes | 2,340 | 1 | yes |
| m7 | 1763 | 8,100 | yes, 1,209,600 codes | 2,700 | 1 | yes |
| m8 | 1333 | 453,600 | yes, 1,209,600 codes | 1,980 | 1 | yes |
| m9 | 1927 | 10,800 | yes, 1,209,600 codes | 2,790 | 1 | yes |
| m10 | 1739 | 10,710 | yes, 1,209,600 codes | 2,520 | 1 | yes |
<!-- /CODES -->

- **The facts leave thousands of codes.** In every fixed-modulus dataset, 8,100–453,600 digit codes pass x < N on every
  training row, and in 12 of the 14 as many again with N read backwards. With any sampled moduli, one code passes.
- **The answers pin the code**, in 12 of the 14: exactly one code also reproduces every training answer the file uses,
  the true one (`code_search.py`, exact arithmetic standing in for an exact machine). M1 and M2 have no answered row
  with T ≤ 2.
- **But the file cannot get there.** Its reader takes its code from facts only, and its subtraction table takes its
  meaning from the same facts, so no machine is exact until the code is; using the answers would mean scoring
  8,100–453,600 whole codes with a machine that cannot yet compute. The law library does not change that (§1: laws on every
  pair leave the table wrong on one modulus), and no mined fact compares two digits that are not N's.

**The menu's boundary: `minus1` (y² − 1), held out.** Subtracting 1 is adding N − 1, not a one-digit constant, so no
combination of the shared menu can compute it. The run did what it should: both lesson sets learned their exact
machines (every label-free check exact from phase C on), but no set-and-round combination reproduced a single answered
row (bar a rare coincidental row, on which the selector and the round options drifted to an arbitrary combination:
below 0.75, +4, × the input), so the run never froze; it trained to the 3,600 s deadline (1,013 steps). Score: 0.04%,
max T 0. This is the honest edge of the method: the answers choose among the hypotheses the designer wrote, and fail
visibly (no answer fits, no freeze) when none is right.

**T's direction: M5 certified on both seeds run (74 and 1, on CPU), by the reader's start, not by a lesson.** On
data whose T always has one digit nothing tells "16" from "61" (LITMUS.md §2). No general lesson can say it either:
there is no fact to mine and no answer that differs, so the general file keeps whatever direction its reader starts
with, as the best model did. Both runs started the right way: every check exact, never frozen (no run on one-digit T
freezes, since the file freezes only once T's direction is settled), cut by the deadline at steps 821 and 827, and
certified T = 64 on both ladders. The best model read T backwards on 2 of its 4 seeds, seed 74 among them (50.5%), and
E3's seed 2 below shows the same coin landing the other way for this file. LITMUS.md's fix was a prior (read T the way
the answer is written), which this study did not add.

**Within the family: E3 and M4 (sampled moduli), a regression check.** The best model learned both unchanged
(LITMUS.md, relaxed budget: 3,600 s, at most 1,500 steps, on H100s). With the general file, E3 on seed 74 (on an
8-CPU container, where the step cap, not the deadline, ended it) did **not** finish within the 1,500 steps: its reader
ranked its readings at step 156, as litmus's run did (163), but then took until step 686 to commit to the (right)
reading (litmus's: 263; both runs' posteriors were 0.962 at step 250, then mine hovered near
0.98 for 400 steps below the 0.99 the reader waits for). The table started late, was at 79% of its checks when the
step cap ended the run, and phase C never began: 1.6%. Nothing the general file adds acts on the reader before it
commits (the miner computes without gradients; every new lesson waits for the reading), so this looks like the reader's
different random start (the extra parameters shift every later initialization) meeting a step cap that litmus's run,
in phase C from step 655, cleared. The runs that followed bear that out, and show two more things, both from the
data's T: E3, M4 and M5 train on one-digit T only (E3 always 2, M4 always 8, M5 2, 4 or 8).
- **E3 seed 74 was slow, not wrong**: with 3,000 steps (Modal ran that job on an H200) every check was exact from step
  1,250 and the run certified T = 64 on both ladders, cut by the deadline at step 2,383. Seed 1 certified on CPU, cut by
  the deadline at step 685 with every check exact, and on an H100 too, cut at step 1,483.
- **T's direction is the reader's coin flip** here too, as on M5 above: no run on these tiers freezes, and each is
  evaluated with the direction its reader started with. E3's seed 2 started backwards, on CPU and on an H100 alike: an
  exact machine, exact up to T = 8 on both ladders, then 22, 22 and 94 of 512 rows right at T = 16, 32 and 64 on seen
  moduli (the tier's own test splits hold only one-digit T, so its score reads 100%).
- **M4's answers never reach the selector.** The selector, like the best model's reading lesson, uses only the answers
  of rows with T ≤ 2 (`REAL_MAX_T`, to keep the 80 machines cheap), and every M4 training row has T = 8. So no
  combination ever fit (0.000 for both sets), the selector and the round options stayed at their random start
  (Montgomery 0.575), and the file answered with Montgomery's machine: 0.8%, on an H100 like litmus's certified run.
  The best model, with one lesson set, had nothing to choose. A general fix would let the selector use the smallest T
  the batch offers; it was not built.

**The frozen-reader diagnostic** (`variants/submission_general_reader_frozen.py`, not a submission: the reader set to
the truth by hand): on E1 the digit lessons started at step 136 with every fact right, and after 1,500 steps the
table had 46 of its 200 entries right (149 borrows, 51 digits; its checks passing 61%), for a score of 4.7%. LITMUS.md's
frozen-reader run on E1, without the laws on every pair, ended with the same 46 entries and the same 4.7%. On M1
(N = 10403, five digits) the same diagnostic ended with 39 entries right (its checks passing 27%) and 2.3%, against
LITMUS.md's 54 entries (72%) and the same 2.3%. That is the full-scale version of §1's tiny result: even with the
reader given, one modulus does not teach the table, and laws on every pair do not change that.

### 5. H1

**The official run: certified.** `gen_h1_official` is the file byte for byte (md5 `ddfdc065…`, the record embeds the
source) through the untouched official manifest `benchmark/manifests/h100_hard_h1.json` on a Modal H100 (seed 74):
- T = 64 on both ladders, every rung 768/768; test 9,999/9,999, ood_t and ood_n_t 10,002/10,002 each;
- training 1,856 s of the 3,600 s budget (frozen at step 645; the frozen steps up to the file's 20,000-step cap are
  nearly free), evaluation 97 s of 1,800;
- the selector ended at 0.9986 on the below set, whose wiring is long division by repeated subtraction (with
  WIDE_ANCHORED.md's kind of harmless alias: two final gated subtractions that never fire); the lowzero set had learned
  a Montgomery reduction by repeated addition, which fits no H1 answer; the round options are the plain square;
- the multiply's layout is the true one (288/288), the subtraction table 200/200, and the audit found no table row,
  product cell, countdown row or gate row used at evaluation that training never used;
- the schedule: reading chosen at step 174 (the best model's official run: 97; the extra parameters change the reader's
  random start), digit lessons 219, phase C 410, the below set's wiring lessons 432, T's direction 466, the lowzero
  set's wiring lessons 471, frozen 645.

**CPU seeds.** The same evaluator with the device set to cpu and the seed changed (74, 1–7), the official 3,600 s
budget, on 8-CPU containers (not the official device). All 8 certified: T = 64 on both ladders, every rung 768/768,
every split exact, the lesson set chosen long division (selector 0.9986–0.9993) with the plain square. The margin is
thin, because phase C runs about twice as slow per step as the best model's on CPU (§7): six seeds froze, at steps
629–750, the last of them (seed 2, at step 644) just as the deadline came, and seeds 74 and 3 reached the deadline at
steps 640 and 586 without freezing, every check exact, and certified anyway. Evaluation took 198–624 s of 1,800. Seeds
3, 4, 6 and 7 are relaunched runs (`wave7_relaunch.json`): Modal preempted the first runs of seeds 3 and 4 at about
04:40 while they were being evaluated (both had frozen, at steps 629 and 748, and were exact on every rung evaluated so
far), and the restarted plan driver's duplicates held up seeds 6 and 7. The best model certified all 8 CPU seeds in
1,913–2,361 s of training (WIDE_ANCHORED.md).

**The gradient test (rule 8).** `local_check.py grad` (on Modal: it needs about 12 GB) with every table and switch at the
truth, on 8 rows of H1's depth_t_8 and depth_ood_n_t_4 (`logs/gradtest.log`):
- below selected (H1's machine; all 8 rows exact): every parameter tensor gets a finite, nonzero gradient from the
  final logits except the unselected lowzero set's 13 switches, which take no part in the prediction;
- below selected with every round extra switched on (×10, +1, × the input): the same, the round's three switches
  included;
- lowzero selected (Montgomery's machine; it answers H1's rows wrong, which does not matter here): besides the unselected
  set's switches, two more get exactly 0, the lowzero set's `start` switch and the round's constant `radd`. Both act at
  the first of the eight Montgomery steps, and the prediction path damps the gradient 10× per link (`GRAD_DAMP`), so it
  underflows on the way. That is the best model's known rule-8 edge (OVERVIEW.md §6, rule 8 (b)), met earlier here
  because Montgomery's reduction brings the dividend in from its lowest digit.

### 6. Rules

The general file keeps every rule status of the best model (OVERVIEW.md §6) and changes these:

| rule | status | why |
|---|---|---|
| 7. no hard-coded algorithm | gray, as before, and wider | the round menu adds three switches (×10, + a constant 0–9, × the input and reduce again) and the lesson sets add a second reduction, both written by hand; the menus still contain the right wirings. What is new is that none of them is chosen by the designer for a task: the answers choose |
| 8. end-to-end, unbroken gradient path | gray, as before | the prediction path is straight-through over the lesson sets, so the final logits' gradient reaches the selector and every switch of the selected set; the unselected set's switches get none (they take no part in the prediction). The best model's gray edges remain |
| 14. no data inspection, augmentation, task-specific solvers | gray, partly better, partly new | better: no lesson states a fact about H1's data; premises are mined, and the step lessons are candidates the answers can reject. New: the miner computes statistics over the training batches inside the model (§2) and the template and the round menu are designed. The model still makes up no numbers, never reads the data files, calls no backward |
| 4. the evaluator owns the loop | gray, as before | the schedule is Python state of the model; the miner's streaks and the selector's joining are more of it |

The two diagnostics in `variants/` set the reader by hand and break rule 6; they are not submissions.

### 7. Budgets

The general file costs more per step than the best model once phase C starts: every lesson set runs its own chain
lesson (9,408 combinations of the chain's switches and the gate), step law (1,700) and reading lessons (256, plus the
answers through the whole machine), and the selector's mixture runs 2 × 40 machines on the answered rows. Seconds to a
step, from the evaluator's progress lines:

<!-- TIMES -->
| run | device | frozen at step | s to step 200 | 300 | 400 | 500 | 600 | 700 | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|
| wide_anchored_official (the best model, H100) | H100 | 588 | 67 | 123 | 195 | 311 | 410 | 411 | 558 | 41 |
| gen_cube_s1 | H100 | 630 | 92 | 201 | 332 | 709 | 1128 | 1260 | 1269 | 101 |
| gen_cube_s74 | H100 | 646 | 54 | 145 | 267 | 527 | 906 | 1079 | 1087 | 88 |
| gen_h1_cpu_seed1 | CPU | 629 | 59 | 160 | 341 | 1278 | 2325 | 2632 | 2857 | 525 |
| gen_h1_cpu_seed2 | CPU | 644 | 87 | 222 | 447 | 1657 | 3056 | - | 3604 | 587 |
| gen_h1_cpu_seed3 | CPU | - | 119 | 292 | 594 | 2170 | - | - | 3613 | 310 |
| gen_h1_cpu_seed4 | CPU | 750 | 48 | 113 | 190 | 359 | 1394 | 2742 | 3583 | 202 |
| gen_h1_cpu_seed5 | CPU | 646 | 88 | 208 | 419 | 1486 | 2721 | 3286 | 3550 | 542 |
| gen_h1_cpu_seed6 | CPU | 646 | 64 | 150 | 302 | 1305 | 2510 | 3074 | 3269 | 198 |
| gen_h1_cpu_seed7 | CPU | 717 | 55 | 124 | 254 | 585 | 1967 | 3231 | 3600 | 507 |
| gen_h1_cpu_seed74 | CPU | - | 97 | 200 | 443 | 1570 | 3038 | - | 3608 | 624 |
| gen_h1_official | H100 | 645 | 87 | 235 | 435 | 795 | 1308 | 1538 | 1856 | 97 |
| gen_minus1_s74 | H100 | - | 76 | 213 | 399 | 768 | 1326 | 1878 | 3603 | 163 |
| gen_mont_s1 | H100 | 633 | 78 | 184 | 312 | 653 | 1039 | 1162 | 1171 | 85 |
| gen_mont_s2 | H100 | 652 | 106 | 226 | 365 | 911 | 1606 | 1811 | 1823 | 94 |
| gen_mont_s3 | H100 | 659 | 135 | 291 | 444 | 780 | 1233 | 1497 | 1509 | 109 |
| gen_mont_s74 | H100 | 698 | 54 | 134 | 240 | 466 | 771 | 1062 | 1068 | 70 |
| gen_plus3_s1 | H100 | 630 | 98 | 222 | 376 | 766 | 1161 | 1273 | 1281 | 53 |
| gen_plus3_s74 | H100 | 646 | 47 | 127 | 235 | 442 | 736 | 872 | 879 | 41 |
| gen_rho_s1 | H100 | 630 | 77 | 174 | 290 | 588 | 914 | 1013 | 1022 | 44 |
| gen_rho_s74 | H100 | 641 | 54 | 150 | 275 | 527 | 868 | 1004 | 1012 | 46 |
| gen_ten_s1 | H100 | 630 | 128 | 257 | 414 | 808 | 1193 | 1313 | 1325 | 60 |
| gen_ten_s74 | H100 | 646 | 74 | 206 | 384 | 712 | 1182 | 1397 | 1406 | 84 |
| gen_tenrho_s1 | H100 | 630 | 75 | 173 | 291 | 584 | 916 | 1015 | 1024 | 45 |
| gen_tenrho_s74 | H100 | 641 | 76 | 217 | 405 | 762 | 1247 | 1445 | 1454 | 90 |
| gen_xfirst_s74 | H100 | 562 | 100 | 204 | 378 | 666 | 846 | 847 | 853 | 39 |
<!-- /TIMES -->

On an H100 the general file trains 853–1,856 s, freezing at steps 562–698 (the best model: 558 s, frozen at 588), well
inside H1's 3,600 s. Per step, from the progress lines: 0.8–1.5 s in phases A–B (the best model: 0.56) and 2.9–5.1 s
late in phase C, steps 500–600 (the best model: 1.0); the H1 official run's container was the slowest of the H100s, and
these steps are launch-bound Python work, so the host matters. The extra cost in phase C is the second lesson set and
the selector's mixture; in phases A–B, both sets' hard checks, the laws on every pair and the miner. Evaluation stays
small (39–97 s of 1,800): at evaluation only the selected set's switches and the chosen round options run.

Cheaper versions were not needed on the H100 and were not built ("do not optimize early"): the tournament's pruning
(TOURNAMENT.md) would stop a lesson set's lessons once its selector weight falls far behind (from about step 470 here,
when the answers decide), which would roughly halve the rest of phase C; the library's routines (LIBRARY.md) were not
used, since the lesson sets' blocks are already the skeleton's subroutines (chain, step, reading).

**On CPU** (8-CPU containers, the official 3,600 s budget, not the official device) the early phases are as fast as on
the H100 (step 300 at about 110–290 s: these phases are many small operations), but phase C runs about 9–16 s a step
(a hundred phase-C steps took 937–1,576 s across the eight seeds), against 5–7 s for the best model's CPU seeds (their
records: steps 400–600; WIDE_ANCHORED.md: 1,913–2,361 s in all). That leaves almost no slack in the 3,600 s budget:
six seeds froze, between about 2,400 s (seed 1) and the deadline (seed 2), and two reached the deadline unfrozen
(seeds 74 and 3, at steps 640 and 586) and certified only because every check was already exact (§5). On CPU, pruning
the losing lesson set once the selector has decided (§8) would matter.

### 8. Open questions

1. **Answers choosing the digit code.** On 12 of the 14 fixed-modulus tiers the answers single out the true code among
   8,100–453,600 candidates (§4). Using that needs a machine that is exact *in the reader's labels* before the code is
   right, which this design does not have: its table's borrows come from the same facts. One route would tie the table's
   borrows to the reader's ranking (a design choice that fixes half the table's meaning by construction), then score
   codes as whole options by the answers, as the selector scores lesson sets. Untested.
2. **Pruning a losing lesson set.** The tournament's pruning (TOURNAMENT.md) would stop the losing set's lessons once
   the selector has decided (about step 500 here), bringing phase C back toward the best model's cost.
3. **A reduction outside the template.** Barrett's reduction, or any reduction whose step establishes another
   predicate, would measure how far the template reaches the way the held-out formulas measure the round menu.
4. **T's direction.** Nothing in one-digit-T data can decide it; a prior (read T the way the answer is written,
   LITMUS.md) would, but it is a designer's choice, not something the answers can select.
5. **When each lesson joins.** The tiny test of §1 shows the schedule matters: the algebraic laws must wait until the
   facts have shaped the table. The general file keeps the best model's designed schedule (`LATE_AFTER`, and the
   stages before it); letting each law join when the table passes its own checks, or letting the answers choose among
   schedules as they choose among lesson sets, is untested.
6. **Which answers the selector reads.** It reads the rows with T ≤ 2 only, inherited from the best model's reading
   lesson (`REAL_MAX_T`), and so it is blind on M4, whose rows all have T = 8. Reading the rows at the smallest T the
   batch offers would cover M4 at about four times the selector's cost there (8 squarings instead of 2); not built.

### 9. Reproduce

The general file is built from the certified file, which is never modified (each patch must match exactly once):

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python make_general.py
```

Tiny cores and the data preview (Modal CPUs, app `general-lessons-tiny`; at most 20 containers):

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run ../cpu_first/general_lessons/modal_tiny.py --plan ../cpu_first/general_lessons/plans/tiny1.json
```

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run ../cpu_first/general_lessons/modal_general.py --mode preview
```

Full runs (app and Volume `general-lessons`; litmus/modal_litmus.py with the held-out tasks added to the image; the
per-launch caps were lowered between launches to keep every launch together at 20 containers or fewer), then the
waiter, then the tables. The plans, in the order they ran: `wave1.json` (H1 official and the litmus and held-out tasks,
H100), `wave2_fixed.json` (the 14 fixed-modulus tiers, relaxed, CPU), `wave3_h1_cpu.json` (H1's CPU seeds),
`wave4_seeds.json` (more seeds of the formula tasks, H100), `wave5a.json` and `wave5b.json` (M5's seeds, E3 and M4 as a
regression check, and the frozen-reader diagnostics, CPU; `wave5_m5_frozen.json` is the two together), `wave6_e3.json`
(E3 on seeds 1 and 2, CPU; its third run, E3 with 3,000 steps on CPU, never started and was stopped: on CPU the 3,600 s
deadline would bind before the steps), `wave7_relaunch.json` (H1's CPU seeds 3, 4, 6 and 7 and M4 again, after Modal
preempted them at about 04:40), `wave8_e3_h100.json` (E3 on seed 74 with 3,000 steps, H100) and
`wave9_h100_regression.json` (E3 on seeds 1 and 2 and M4 on H100s, as litmus ran those tiers). At that preemption the
plan drivers of waves 3 and 4 were preempted too, and Modal restarted each with its input, so each spawned its whole plan
again; those duplicate runs were stopped, and the driver now re-attaches to the calls it spawned (their ids kept on the
Volume) instead. That was exercised once: wave 8's driver was preempted at 05:44 and re-attached to its running job
without spawning it again:

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/general_lessons/modal_general.py --mode plan --plan ../cpu_first/general_lessons/plans/wave1.json
```

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python wait_fetch_general.py gen_h1_official
```

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python summarize_general.py fill LESSONS.md
```

The gradient test (§5; on Modal, three containers):

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run ../cpu_first/general_lessons/modal_tiny.py --grad
```

The fixed-modulus code search (§4) and the data preview (§2) run in the competition image:

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run ../cpu_first/general_lessons/modal_general.py --mode codes
```

Local plumbing checks (each under a minute and well under 1 GB; the gradient test is not: it took 11.6 GB once and
belongs on Modal):

```bash
cd ~/github/one-layer/cpu_first/general_lessons && ~/github/sotaku/venv/bin/python local_check.py ../../one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2 N=4
```

### 10. Files

In `cpu_first/general_lessons/` (new; no other file was edited, and the certified files' md5s match
`logs/md5_at_start.txt`):
- `LESSONS.md`: this write-up;
- `submission_general.py`: the general file (built by `make_general.py` from `wider_menus/submission_wide_anchored.py`);
- `variants/submission_general_reader_frozen.py`: the diagnostic with the reader set to the truth (not a submission);
- `tiny_general.py`, `tiny_table.py`: the tiny cores of §3 and §1; `modal_tiny.py` runs them; results in `tiny/`;
- `mine_preview.py`, `preview.json`: §2's data-side facts and code counts; `code_search.py`, `codes.json`: §4's
  brute-force code search on the fixed-modulus tiers;
- `tasks_general.py`: the held-out tasks (litmus/tasks.py's generators reused, three answers added);
- `make_general.py`: builds the general file and the diagnostic variant from the certified file;
- `modal_general.py`, `wait_fetch_general.py`, `summarize_general.py`, `local_check.py` (and `gradtest_data/`, the 8
  H1 rows of the gradient test);
- `plans/`, `records/` (every full run: log, result, manifest, exact source), `logs/`.

One correction to the general file's own docstring, which is left as certified: it says the final logits' gradient
reaches "the selector and both sets' switches"; the gradient test (§5) shows it reaches the selector and the selected
set's switches, not the unselected set's.
