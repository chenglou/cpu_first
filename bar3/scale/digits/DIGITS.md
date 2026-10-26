# One more digit: bar 3's learner with x as digits

<!-- PLAIN -->
## In plain words

**The question.** Bar 3's learner counts. A loop repeats its body x times, so a program over x takes about x steps,
and at x of 6–8 digits that is past any budget. This study gave the learner two things:
- x as digits: one new loop, `D.B[...]`, which runs its body once per digit of x in base B, with the digit in a
  register;
- one new source, which asks: is the answer for x·B + d a function of the answer for x, the digit d and N? If so,
  it poses that step as a practice problem.

Three targets, x mod N, x² mod N and 2^x mod N, each with x counted (unary), in base 2 and in base 10. Both rules,
with and without the source: 36 runs, 5 rounds each, from scratch.

A program counts only when it is right on the target's rows and on 200 values of x with 6–8 digits within 100,000
steps. The rows are x below 70 at N ≤ 7 and below 100 at N ≤ 10. No loop over the units of x can meet that budget.

**What came.** x mod N in base 2, with the source, under both rules: in round 1 under anchored pairs, in round 3
under hardest first.
- Both programs loop over x's binary digits, with the step the source posed as the body: X ← d + 2r, then
  "subtract N if you can".
- That is long division's step for base 2 (Horner's rule, with the quotient thrown away).
- Each is right on 1,000 values of x below 10⁹ at N ≤ 10, and on 1,000 at N ≤ 40.

Nothing else came. Without the source nothing poses a step that takes a state and a digit, so no control run came.

| target | unary | base 2 | base 10 |
|---|---|---|---|
| x mod N | no | with the source: round 1 (anchored pairs), round 3 (hardest first); without it: no | no |
| x² mod N | no | no | no |
| 2^x mod N | no | no | no |

**Why the others did not.**
- **Unary.** Its step, (r + 1) mod N, is the cheapest of all (cost 3). But every program right on the rows is a loop
  over the units of x, and it runs out of steps at big x.
- **Base 10.** The step is ×10, then up to nine subtractions. The probes never found it.
- **2^x.** The binary step is square-and-multiply's, t ← t²·2^d mod N. It needs a squaring and a doubling mod N, and
  was never learnable in 5 rounds.
- **x² mod N.** In round 0 of all 12 runs the subgoal's score found the hidden state: x mod N on long x. From round 1
  the learner solved it, but its alias check called it a duplicate of the law book's x mod N on short pairs, a
  problem on other inputs that the same program solves.
  - Under hardest first, and under anchored pairs in unary, that twin was never picked, so no step was posed from
    it.
  - Under anchored pairs in base 2 and base 10 the twin was practised in round 1, and the source posed its step from
    round 2. In base 2 that step was learnable.
  - From round 3 the twin and x² mod N on the pairs were routines, so `D.2[step] MOV.X.A square` was one pick away.
    But this rule scores how much shorter the target's rows become, and counting programs right on every row had
    already made them short.

**Does working in log space make the step easier to find, and to extrapolate?**
- **Easier to extrapolate: yes.** Only a loop over digits meets the budget, and the step found on two-digit x is
  right at nine digits.
- **Easier to find: no.** The step gets longer as the base grows: 3 tokens in unary, 4–5 in base 2, out of reach in
  base 10. Base 2 pays the least, as predicted.

**Against the predictions.**
- All five general predictions held:
  - the pose rule's truth table;
  - unary never comes;
  - nothing comes in base 10 that base 2 lacks;
  - no digit loop right on the rows fails at big x (the programs that failed were loops over units);
  - x² mod N's round-0 subgoal is x mod N on the rows.
- In the run table, 35 of the 36 runs came out as predicted. Hardest first's binary x mod N came in round 3, one round
  after the predicted rounds 1–2.

**Against stage 3.** Stage 3 gets long division's step from a designed step law, on a machine with an instruction
that brings a digit in. Here the step came from the target's rows and a generic loop over digits. This tests only
the outer loop. At N ≤ 10 the step is one subtraction; at H1's N, up to 2²⁴, the step must itself be a digit
machine.

**Cost.** $50.69 billed by 02:05 UTC, of the soft $80; the report's last hour is still to come. 446 search-hours.
At least 17 preemptions, each replayed from the stores.

**What it says.** Given a loop over digits and a source that asks one question of the rows, the learner invents long
division's step for base 2 and the loop around it. The result is right from two-digit x up to nine digits. What
stopped the rest:
- the learner's bookkeeping: an alias check that hides the hidden state behind its twin, and a score that sees the
  rows, not the check at big x;
- the probes' reach: base 10's step, and square-and-multiply.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

`PREREG.md`, written 2026-10-25 09:40 PDT before any learner run and before any search on this study's problems,
read-only since (sha256 `cf6a37167535b40e0ae0d09d3ea55530af83af202875ab6db74a4cdf0dc6dbee`; its written time was
corrected from 09:45 to 09:40 before anything ran). It holds the question, the machine, the learner's changes, the
targets, the check at big x and its budget, the run plan and the predictions; §6 marks each. PREREG.md says this section
repeats its text unchanged; it does not, so read PREREG.md itself.

### 2. What was built

#### 2.1 The machine: one loop over a number's digits (`machine.py`)

`D.B[...]` reads X once on entry, as `L.X` does, and runs its body once per digit of that value in base B, from the
top, with the digit in X. B = 1 is unary: v ones, so `D.1` is `L.X` with X set to 1 each time. The value 0 has no
digits. Every iteration costs a step as it starts, as `L.r`'s do.

It is a design choice, and a small one:
- it knows nothing of remainders or of arithmetic, and multiplies by nothing;
- it walks one number's digits, the way `L.X` walks its units, and leaves the digit in the register the input
  arrives in.

Stage 3's frame (`library/dsl.py`, INTEGRATION.md) is far more specific to digits:
- registers of digits, a shared pointer, a borrow, and cells at the pointer (subtract-with-borrow, add-with-carry,
  multiply-accumulate);
- loops by the number's length and by the base;
- `BRG.X`, "bring X's top digit in", which does A ← A·B + X's top digit. That is half of Horner's step, built into
  the instruction set.

Here the ×B and the + d are the learner's to find.

Added to brm's interpreter and to the compiled search (`bar3/scale/engine.py`'s compiler and kernels), in the
processes of this study only. Two checks:
- on 3,000 random programs, half with the block, the kernel and the interpreter agree on every state and death;
- without the block, the kernel gives the original's states, deaths and hash keys.

#### 2.2 The data: the target's rows and the check at big x (`targets.py`)

The target's rows are every (x, N) with N ≤ k and x < 10·k: x of two decimal digits and up to 7 binary ones, with
every residue among the numbers that get a digit appended. That is 490 rows at the search size (k = 7) and 1,000 at
the check size (k = 10).

The check at big x: N ≤ 10, 20 values of x in [10⁵, 10⁸) per N (seeded), a cap of 10⁹ (so X holds x) and 100,000
steps per instance. A loop once per unit of x takes at least two steps per unit, so at least 2·10⁵ at x ≥ 10⁵: it
cannot pass. §3.2 shows both kinds of loop on hand-written programs.

#### 2.3 The source: one more digit (`source.py`)

The source reads the data's rows only. For each N, every x whose B children x·B + d (d < B) are all among the rows
gives a row: (answer at x, d, N) → answer at x·B + d. In unary, the one child is x + 1, with d = 1. The answer is its
own state when those rows are a function at both sizes.

What it poses:
- **The states.** The target's answer, and what each routine of the library leaves in A on the target's inputs.
  When the target's answer alone is not a state, also a pair: the target's answer with one routine's (x² mod N is
  not its own state; with x mod N beside it, it is).
- **The placements.** A state is placed in A, U or T, updated in place, with the digit in X, U or T. A pair goes in
  two of A, U and T, with the digit in X, where the digit block puts it. A step for one state comes in 7 placements.

#### 2.4 The learner (`learn.py`, written by `make_learner.py`)

It is the held-out test's frozen learner, with each change applied as an exact replacement, the same in every arm:
- the digit block of the representation in the loop kinds;
- the target posed on its rows, and everything posed from the target's rows with it. The recurrence source still
  reads the pairs x < N;
- the law book posed on the target's rows too (X = x, depth 1), besides the pairs x < N;
- a program for the target accepted only when it is right on all its rows at both sizes and passes the check at big
  x, on every search on the target;
- the target's machines given more steps (20,000 and 100,000).

The arm `src` adds the source, every round. Configs (`configs.py`): `<rule>_<u|b|d>_<src|ctl>`, for hardest first
(`lp_clean`) and anchored pairs (`ct2a2_clean`), on the frozen budgets.

### 3. The tiny tests and the checks (local; `tiny.py`, `check.py`, `smoke.py`)

#### 3.1 The pose rule's truth table (before PREREG; `logs/tiny_table.txt`)

This is the theory note's cheap test 1: the source's test alone, with no search, on the note's twelve targets, in
unary, base 2 and base 10, on two sets of rows (the learner's own, x < 70 at N ≤ 7 and x < 100 at N ≤ 10; and the
note's, every x < 2,000 at N ≤ 7). There were **0 disagreements with §5.2**:
- the answer is its own state for x mod N, 2x + 1 and 2^x;
- it is not for x², x² ± 1, x² + 3, 10·x², 10·x² + 1, x³, Montgomery's or Fibonacci;
- the same in all three representations, on both sets of rows.

#### 3.2 The digit block and the budget, on hand-written programs (before PREREG; `logs/tiny_programs.txt`)

`T.mod` is bar 3's link, "subtract N if you can". Every program here is right on all 490 and 1,000 rows. Steps are
the most over the 200 big-x instances, and the count at x = 10⁵ (N = 7):

| program | big x | steps |
|---|---|---|
| x mod N, counting: `L.X[INC.A MOV.X.A T.mod]` | fails (steps) | 2,585,715 at x = 10⁵ |
| x mod N, unary: `D.1[INC.A MOV.X.A T.mod]` | fails (steps) | 2,585,715 at x = 10⁵ |
| x mod N, binary: `D.2[L.A[INC.X] L.A[INC.X] T.mod]` | **passes** | 1,401 at most |
| x mod N, decimal: `D.10[` ×10 written out, `T.mod`, reduce `]` | **passes** | 15,760 at most |
| x² mod N, binary: `D.2[…]` then square and reduce | **passes** | 4,234 at most |
| x² mod N, decimal | **passes** | 17,245 at most |
| 2^x mod N, counting and unary (doubling x times) | fail (steps) | 2,933,356 at x = 10⁵ |
| 2^x mod N, binary: `INC.A D.2[square, double if the bit is 1]` | **passes** | 42,412 at most |

This sized the budget at 100,000 steps: twice the costliest digit loop, and half of what any loop over units needs.

The frozen check machine (6,000 steps at k = 10) kills three of these correct programs on the check rows. Binary 2^x
is one of them. So the target's machines got more steps (§2.4).

#### 3.3 What the source poses (after PREREG; `logs/tiny_pose.txt`)

In every base, the source poses 7 placements of the step for x mod N and for 2^x from round 0, since each target is
its own state. Each comes with its rows:
- x mod N: 28 at k = 7 in unary, 56 in binary, 280 in decimal;
- 2^x: 18, 36 and 180.

For x² mod N it poses nothing while the library is empty.

#### 3.4 The learner's probes on the posed steps (after PREREG; `logs/tiny_probe.txt`)

These are the frozen 30k-state and 1M-state probes, with "subtract N if you can" as the one routine (the library
after round 0 in every held-out run). For x² mod N, x mod N on the rows (`MOV.A.X L.A[MOV.X.A T.mod]`) was a routine
too.

| base | x mod N's step | x² mod N's (x mod N's values) | 2^x's step |
|---|---|---|---|
| unary | cost 3, `MOV.X.A INC.X T.mod` | cost 3 | cost 4, `L.A[INC.A] MOV.X.A T.mod` |
| binary | **cost 4**, `L.A[INC.X INC.X] T.mod` | cost 4 | not within 1M states |
| decimal | not within 1M states | not within 1M | not within 1M |

The binary step is X = d + 2A, then "subtract N once". It is a token shorter than PREREG expected (cost 5).
- `D.2[step]` is right on all 1,000 rows and at big x.
- `D.1[step]` is right on the rows and fails at big x, on steps.

#### 3.5 The paths the smoke run did not reach (`logs/tiny_accept.txt`)

- With the binary step as a routine, the target search accepts `D.2[T.step]` at cost 2 (general 1.0), in 0.01 s.
- In unary, with the unary step as a routine, it accepts nothing. 657 programs are right on the rows and fail at big
  x, such as `L.X[T.step]`, and the search runs to its budget (16M states, 131 s).
- For x² mod N, with x mod N on the rows as a routine, the source poses that routine's step in 7 placements and the
  pair (x² mod N, x mod N) in 6. The pair's step is not found within the 30k-state probe.

#### 3.6 The smoke run (`smoke.py`; `logs/smoke_*`)

Three configs ran 2 rounds on x mod N in binary, at a tenth of the budgets, on 4 local workers: no job failed. At
those budgets "subtract N if you can" (cost 6) is not solved in round 0, so the step stays out of reach. That is a
property of the small budgets, which the full runs do not share.

#### 3.7 Committed records still reproduce (`check.py`; `logs/check_heldout.txt`)

With this study's code installed (the block in the interpreter and the kernels, the jobs), all 20 held-out records
(committed) were replayed on their stores, and each replay gives back its record. Of their searches, 791 cheap ones
(one in 40 of the probes, looks and target probes) were run again on the patched engine. All 791 match their stored
records in every field but the seconds.

### 4. The runs (`modal_digits.py`; `records/*_dg1.jsonl`; `logs/`)

36 runs:
- 3 targets × 3 representations × 2 rules × 2 arms;
- from scratch, 5 rounds, seed 0, the frozen budgets;
- in six Modal apps, one per rule and representation, launched 2026-10-25 16:59–17:01 UTC;
- each app one container (16 CPUs, 64 GiB), its six runs as threads, the searches on a pool of 16 worker
  processes, every finished search stored on the Volume `bar3-digits`.

The containers were preempted often, at least 17 times (§8). A preempted container restarts and replays
its runs from the store. A search running at the time fails, is never stored, and is searched again in full. Every
final record comes from one container's launch, and none holds a failed search.

The spend watcher (`watch_spend.py`, one process for all six apps) would stop an app at $15 and all at $90. It never
had to.

Cost:
- billed by 02:05 UTC: $50.69 for the six apps. The report lags about an hour, and the anchored-pairs apps ran into
  the 01:00 hour, which it does not show yet;
- 446.4 search-hours (`cost.py`; `logs/cost_dg1.txt`);
- from launch (17:00 UTC) to stop: hardest first's groups 4.6–6.8 hours, anchored pairs' 8.4–8.8.

### 5. Outcomes (`verify.py`; `logs/verify_dg1*.txt`)

| target | representation | hardest first, with the source | hardest first, without | anchored pairs, with | anchored pairs, without |
|---|---|---|---|---|---|
| x mod N | unary | no | no | no | no |
| x mod N | binary | **round 3** | no | **round 1** | no |
| x mod N | decimal | no | no | no | no |
| x² mod N | unary, binary, decimal | no | no | no | no |
| 2^x mod N | unary, binary, decimal | no | no | no | no |

"No" is 5 rounds without a program that is right on the target's rows and passes the check at big x.

#### 5.1 The two programs

Both loop over x's binary digits, with long division's step for base 2 as the body. They were found by the plain
target search, from routines the learner practised (`mechanism.py`; `logs/mechanism_dg1.txt`):

- **Anchored pairs, round 1: `D.2[T.dig7]`.**
  - T.dig7 = `L.U[M1.2] T.mod MOV.U.A`, with the merge M1.2 = `INC.X INC.X`.
  - In words: X ← d + 2U, then "subtract N if you can" into A, and U ← A.
  - It is the step the source posed with the state in U and the digit in X (`digit/target/U.X`), practised in
    round 1 at cost 5.
- **Hardest first, round 3: `D.2[MOV.T.X T.dig15]`.**
  - T.dig15 = `L.A[INC.T INC.T] MOV.X.T T.mod`: T ← d + 2A, X ← T, then "subtract N if you can".
  - It is the step with the state in A and the digit in T (`digit/target/A.T`), practised in round 3 at cost 5.
    The target moves the digit into T first.
- **In both,** `T.mod` is "subtract N if you can", from the law book (law/mod(X,N)/x+qN), practised in round 0 at
  cost 6.

They are 11 and 12 primitive instructions. Each is right on all 490 and 1,000 rows and on the 200 instances of the check
at big x. Rechecked on fresh samples:
- 1,000 values of x below 10⁹ at N ≤ 10, in 100,000 steps: right on all 1,000;
- 1,000 at N ≤ 40, with up to 10⁷ steps: right on all 1,000.

#### 5.2 What the source posed, and what the rules did with it

- **x mod N, binary.** The step was posed in 7 placements from round 0. It was learnable from round 1 under both
  rules: A.X at cost 4 (`L.A[INC.X INC.X] T.mod`), and A.U, A.T, U.X and T.X at cost 5.
  - Anchored pairs scores each learnable problem by how much its routine shortens the target. It picked three
    placements in round 1 (A.X, U.X, T.X), and the target followed in that round's search, over U.X.
  - Hardest first takes the four costliest learnable problems, ties in pool order, and the digit source comes last
    in the pool. In rounds 1–2 the cost-5 problems ahead of the steps filled the four picks. In round 3 two steps
    were picked (A.U, A.T), and the target followed.
- **x mod N, decimal.** Posed in 7 placements every round, never learnable in 5 rounds. The step is ×10 and then up
  to nine subtractions, past the 1M-state probe.
- **x mod N, unary.** The step (r + 1) mod N was learnable from round 0 (cost 5), and from round 1 at cost 3
  (`MOV.X.A INC.X T.mod`). Hardest first never picked it: it was too cheap. Programs right on the rows were plenty,
  e.g. `L.X[T.mod MOV.X.A]` (x mod N by counting), and all fail the check at big x on steps.
- **2^x mod N.**
  - Unary: the doubling step was learnable from round 1 (cost 4). Hardest first practised three placements in round
    4. The programs built on them loop over units (`INC.T L.X[T.dig19] MOV.X.T M0.3`), right on the rows and failing
    at big x on steps.
  - Binary and decimal: never learnable. The binary step t ← t²·2^d mod N needs a squaring and a doubling mod N,
    past the probes with these libraries.
- **x² mod N.**
  - In every representation and under both rules, round 0's subgoal is x mod N on the rows (law/mod(X,N)/rows), as
    predicted. The table score sees that x² mod N is a function of it and N.
  - Its deeper probe (10M states) fails in round 0, before "subtract N" is a routine. From round 1 the probes solve
    it at cost 3 (`L.X[T.mod MOV.X.A]`, later 2).
  - The learner's alias check then calls it a duplicate of law/mod(X,N)/x+QN, the law book's x mod N on the pairs
    X = x + qN: the same function, posed on other inputs. A duplicate is never "learnable", so the subgoal is never
    practised.
  - Its twin is cheap (cost 2–3): hardest first never takes it.
  - So under hardest first no routine holds x mod N on long x, and the source posed no step for x² mod N.
  - Anchored pairs practised the twin in round 1 in base 2 and base 10, though not in unary. From round 2 the
    source posed its step and the pair's (x² mod N with x mod N).
    - In binary the twin's step was learnable from round 2 (cost 4–5, down to 2 by round 4). Its best look gained
      23.3 bits in round 2, below the four picks; x² mod N on the pairs, at 44.8 bits, was one of them.
    - From round 3 the twin and x² mod N on the pairs (T.law10) were routines and the step was learnable, so
      `D.2[step] MOV.X.A T.law10` was one pick away. Yet a look with the step gained −0.1 bits.
    - The looks score how much shorter the target's rows become. Counting programs right on every row (`L.X[T.mod
      MOV.X.A]` and a square) had already made them short, and the score does not see the check at big x.
    - The pair's step was never learnable. In decimal nothing posed was learnable.
- **Without the source** nothing poses a step with a state and a digit. No control run came.

### 6. Predictions against outcomes

| prediction (PREREG.md) | confidence | outcome |
|---|---|---|
| 1. the pose rule's truth table | held before | 0 disagreements (§3.1) |
| 2. unary never comes, either arm | high | **held**: none of the 12 unary runs came |
| 3. base 2 first: nothing comes in decimal and not in binary; binary no later | medium | **held**: nothing came in decimal |
| 4. extrapolation free: every digit-loop program right on the rows passes at big x | medium-high | **held**: both programs pass; of the programs right on the rows that failed at big x, none had a digit loop of base ≥ 2. All were loops over units |
| 5. x² mod N's round-0 subgoal is x mod N on the rows, in every representation, both rules; not for 2^x | medium-high | **held**: all 12 x² mod N runs; none of the 12 on 2^x |

The run table:

| target | arm | predicted | outcome |
|---|---|---|---|
| x mod N | unary, both arms | no (high) | no |
| x mod N | binary, with the source | rounds 1–2 (medium) | anchored pairs round 1: **held**; hardest first round 3: **one round late** |
| x mod N | binary, without | no (medium) | no |
| x mod N | decimal, with | no (low) | no |
| x mod N | decimal, without | no (medium) | no |
| x² mod N | unary | no (high) | no |
| x² mod N | binary, with | no; if it comes, rounds 3–4 (low) | no |
| x² mod N | binary, without; decimal, both arms | no (medium) | no |
| 2^x mod N | unary | no (high) | no |
| 2^x mod N | binary, with | no (low) | no |
| 2^x mod N | binary, without | no (medium) | no |
| 2^x mod N | decimal, both arms | no (high) | no |

The binary step was predicted at cost 5 (`L.A[INC.X] L.A[INC.X] T.mod`). It came at 4 in its A.X placement
(`L.A[INC.X INC.X] T.mod`) and at 5 in the others. The routes PREREG.md gave for x² mod N and 2^x stopped earlier
than it said:
- x² mod N's first piece, x mod N on the rows, was never practised;
- 2^x's binary step was never learnable.

### 7. What it says

#### 7.1 Does working in log space make the step easier to find, and to extrapolate?

The same learner, budget and source, across three representations:

| representation | the step for x mod N | found as | extrapolates |
|---|---|---|---|
| unary (`D.1`, counting) | (r + 1) mod N | cost 3 from round 1 | no: every program right on the rows fails at big x on steps |
| base 2 (`D.2`) | (2r + d) mod N: X ← d + 2r, subtract N once | cost 4–5 from round 1 | **yes**: right on every x below 10⁹ checked |
| base 10 (`D.10`) | (10r + d) mod N: ×10, then up to nine subtractions | not in 5 rounds | – |

- **Easier to extrapolate: yes.** Only a loop over digits meets a budget that counting cannot meet at 6–8 digits.
  The step found on two-digit x is right at nine digits. With N ≤ 10 the rows hold every (state, digit) pair, so a
  step right on them is right at every length (§7.2).
- **Easier to find: no.** The step gets longer as the base grows: unary's is the shortest, base 2's a token or two
  longer, base 10's out of reach. Log space buys extrapolation with a longer step, and base 2 pays the least. That is
  the order PREREG.md predicted.

#### 7.2 The theory note, result by result (`theory/LARGE_NUMBERS.md`)

- **Horner's automaton, Myhill–Nerode (§2.2): tested and rediscovered.**
  - The source's test is the classes condition for one appended digit: equal answers must stay equal after the
    digit.
  - On the note's twelve targets it gives §5.2's verdict exactly (§3.1).
  - The learner found x mod N's automaton in base 2, r ← (2r + d) mod N, as a loop over digits. That is long
    division with the quotient thrown away, at N ≤ 10.
- **Cobham, automatic sequences (§2.4): tested.** The verdict is the same in all three representations, and the base
  changed the step's length, not whether a step exists: cost 3, 4–5, and out of reach.
- **Square-and-multiply (§2.5): not rediscovered.** 2^x's binary step needs a squaring and a doubling mod N, and was
  never learnable in 5 rounds. The note's step is right; it is past these probes.
- **Uniform programs extrapolate (§3.4), the characteristic sample (§3.2–3.3): tested, held.**
  - Both digit loops are right far past the sizes searched.
  - No digit loop that was right on the rows failed at big x.
  - The programs that failed were loops over units.
- **Levin search, cost exponential in length (§3.1): consistent.** Base 10's step is the longest and the one not
  found.
- **§5.2's route for the polynomial targets: not completed.** The score found the hidden state, x mod N on long x,
  as the note's cheap test 2 expects.
  - Its alias check counts it as the law book's x mod N on short pairs, which hardest first never picks.
  - Anchored pairs practised that twin, in base 2 and 10. There the last piece, the step, lost on a score that
    measures the rows (§5.2).
  - Nothing here contradicts the note. What stopped x² mod N was the learner's bookkeeping and scoring, not the
    theory's step.

#### 7.3 Against stage 3's long division

- **Stage 3** learns long division's step from a practice problem made from the template's step law (OVERVIEW.md
  §4, "the template's step laws (now practice problems)"). Its machine has `BRG.X`, A ← A·B + X's top digit, as an
  instruction. The step's shape, r·B + d then reduce, is given twice: as the law and as the instruction.
- **Here** the source poses the step from the target's rows alone. It asks whether the answer at x·B + d is a
  function of the answer at x, d and N, and when it is, poses that function. Nothing names B, Horner or remainders.
  The machine's only addition is a generic loop over a number's digits. The ×2 and the + d came from search
  (`L.U[INC.X INC.X]`: X ← d + 2U).
- **What this tests: only the outer loop.** Here N ≤ 10, the state is one number below N, and the step is computed by
  counting (2r + d < 2N, then one subtraction). In H1, N runs to 2²⁴ and the state is itself a number of several
  digits. Stage 3's step must compute (r·B + d) mod N with a digit machine, one quotient digit at a time
  (LARGE_NUMBERS.md §2.2). Whether the learner can find that inner machine is not asked here.

### 8. Dead ends and loose ends

- **x mod N on long x, a duplicate.** The alias check compares programs by what they do to random machine states. It
  does not tell a problem posed on long x from the law book's on short pairs when one program solves both. The
  subgoal's priority does not pass to the twin. The fix would be to let the subgoal's twin inherit the subgoal (or
  practise the subgoal whatever its alias). Not tried: the learner is frozen here.
- **Hardest first and cheap pieces.** The binary step waited two rounds behind cost-5 problems earlier in the pool,
  and the unary steps were never picked. The held-out test saw the same (HELDOUT.md §7, 2^x).
- **Preemptions.** At least 17 container exits appear in the captured logs: hardest first unary 3, binary 2,
  decimal 0; anchored pairs unary 2, binary 5, decimal 5 (UTC times in `logs/app/`). These are lower bounds. A
  capture holds only the log's tail, and hardest first decimal's "group done" time implies a restart that no
  capture shows. Each one cost the work in flight and a replay, and
  stretched the anchored-pairs groups to 6–7 hours. The records are whole (§4).
- **The spend watchers.**
  - The first six, one per app, were broken by zsh's word splitting (`logs/broken_watchers/`).
  - The next six, one per app (`logs/watch_dg1_<rule>_<rep>.log`), hit the billing report's rate limit together,
    and one crashed.
  - They were replaced by one watcher for all six apps (`logs/watch_dg1.log`). It crashed once, at 23:35 UTC, when
    `modal app list` stopped listing a stopped app. It was fixed (an unlisted app counts as stopped) and restarted a
    minute later.
  - Its estimate for the hour under way keeps adding cost to apps that have stopped, which only makes it stop
    earlier.
- **The probe test's first draft** built its jobs with `bar3.learner.job`, which drops a shaped problem's registers.
  Its probes were then wrong. It was fixed before any result was used (`bar3.scale.learn.job`).
- **The smoke run** was too small to solve "subtract N" in round 0, so it never reached the step (§3.6). The
  paths it missed were checked directly (§3.5).

### 9. Files and how to rerun

Python with numba: the scratchpad venv that ran this study (`nvenv`); the plain one: `~/github/sotaku/venv/bin/python`.

- **Machine and data:**
  - `machine.py`: the digit block, in brm's interpreter and the compiled engine, and its check;
  - `targets.py`: the targets, their rows and the check at big x.
- **The source and the learner:**
  - `source.py`: the source, and the pose rule's truth table;
  - `make_learner.py` writes `learn.py` (11 asserted replacements of the frozen learner);
  - `jobs.py`: the jobs on the target, with the check at big x;
  - `configs.py`.
- **Tests:**
  - `tiny.py`: `table`, `programs`, `pose`, `probe`;
  - `smoke.py`;
  - `check.py`: committed records replay with this study's code.
- **Runs:**
  - `modal_digits.py`, with `searcher.py` (the pool's workers install the block);
  - `watch_spend.py`: one spend watcher for many apps.
- **Results:** `verify.py` (§5), `cost.py` (§4).
- **Records:** `PREREG.md` (read-only); `records/*_dg1.jsonl`, and `records/calls_dg1.json` (the Modal call ids).
- **Logs:** `logs/` holds every output above, the six apps' captured logs (`logs/app/`) and the watchers' logs.

```bash
cd cpu_first && <python with numba> -m bar3.scale.digits.tiny table
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.digits.check STORE.jsonl ... -- RECORD.jsonl ...
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/digits/modal_digits.py --fresh lp_clean_b_src,lp_clean_b_ctl --targets dxmod,dxsq,dpow2 --tag dg1
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/digits/watch_spend.py 15 90 APP_ID ...
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/digits/modal_digits.py --fetch
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.digits.verify
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.heldout.mechanism bar3/scale/digits/records/lp_clean_b_src_dxmod_dg1.jsonl bar3/scale/digits/records/ct2a2_clean_b_src_dxmod_dg1.jsonl
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/digits/cost.py
```
