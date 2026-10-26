# The digits study: pre-registration

Written 2026-10-25 09:40 PDT, before any learner run and before any search on this study's problems. Run before
it, with no search: the machine's check (`machine.py`), the pose rule's truth table (`tiny.py table`) and the
hand-written programs that size the budget (`tiny.py programs`). After it come the tiny tests that search
(`tiny.py pose` and `probe`), a local smoke run, then the runs. DIGITS.md §1 repeats this text unchanged.

## The question

Bar 3's learner counts: its loops repeat a body x times, so a program over x takes about x steps. The study gives it
x as digits and a source that asks, across sizes, whether the answer for x·B + d is a function of the answer for x,
the digit d and N. If so, that step is long division's (x mod N) or square-and-multiply's (2^x mod N), the same at
every length. The theory note (`theory/LARGE_NUMBERS.md`, §5.2) predicts which targets have that step and that base
2 comes first. The user's question: does working in log space make the step easier to find and to extrapolate?
x is given three ways: counted (unary), in base 10 and in base 2.

## The machine (`machine.py`)

One new block, `D.B[...]`: read X once on entry, as `L.X` does; run the body once per digit of that value in base B,
from the top, with the digit in X (B = 1: v ones). It knows nothing of remainders or of arithmetic. In unary it is
`L.X` with X set to 1 each time. Checked: on 3,000 random programs, the compiled kernel and the interpreter agree, and
without the block the kernel is the original's.

A design choice, next to stage 3's frame (`library/dsl.py`): that machine has registers of digits, a pointer, a
borrow and per-place cells, loops by the number's length and the base, and `BRG.X` ("bring X's top digit in", which
multiplies by the base itself). This block only walks one number's digits; the ×B and the + d are the learner's to
find.

## The learner (`learn.py`, written by `make_learner.py` from the frozen one)

The held-out test's frozen learner (`bar3/scale/subgoal/learn.py`, `lp_clean_deep_sub` and `ct2a2_clean_deep_sub`),
with these changes, the same in every arm:
1. The loop kinds include the digit block of the representation (`D.1`, `D.2` or `D.10`) from round 0.
2. The target is posed on its rows, x < 10·k for every N ≤ k (490 rows at k = 7, 1,000 at k = 10), and so is
   everything the learner poses from the target's rows. The recurrence source reads the pairs x < N, as before.
3. The law book is posed on the target's rows too (X = x, depth 1), besides the pairs x < N. On the pairs it poses
   "subtract N if you can", the link; on the rows its mod(X, N) is x mod N on long x, the state the polynomial
   targets need beside their answer.
4. A program for the target counts only when it is right on all the target's rows at both sizes and passes the
   check at big x. This holds on every search on the target: the probe, the looks, the target search and the
   guided search.
5. The target's machines have the caps as before and more steps: 20,000 at k = 7 (3,450 before) and 100,000 at
   k = 10 (6,000 before). A digit loop multiplies a step's cost by the number of digits.

With the source (arm `src`): the one-more-digit source (`source.py`) every round, in the base of the representation.
It runs on the target's answers and on each routine's values on the target's inputs, alone, and as a pair with the
target's answers when the target's alone is not a state. Each step is posed with the state in A, U or T (updated in
place) and the digit in X, U or T. A pair is placed in two of A, U and T, with the digit in X.

The control (arm `ctl`): the same without the source.

## Targets, the check at big x and the budget

- x mod N, x² mod N, 2^x mod N (`targets.py`: `dxmod`, `dxsq`, `dpow2`), each in unary (`D.1`), binary (`D.2`) and
  decimal (`D.10`).
- The stretch target, x^(2^T) mod N, is not run: it needs T as a third input.
- The check at big x: N ≤ 10 and 20 values of x in [10⁵, 10⁸) per N (seeded), with a cap of 10⁹ and 100,000 steps per
  instance. A program passes when it is right on all 200. A loop once per unit of x takes at least 2 steps per unit,
  so at least 2·10⁵ at x ≥ 10⁵, and cannot pass. The hand-written digit loops take 1,401 (binary x mod N) to 42,412
  (binary 2^x) steps (`logs/tiny_programs.txt`).
- Runs: 3 targets × 3 representations × 2 rules × 2 arms = 36, from scratch, 5 rounds, seed 0, the frozen budgets.
- Money: soft $80, cap $100; every launch guarded by `spend.py 4750`.
- "Came in round r" means the plain or guided target search accepted a program in round r.

## Predictions

The theory note's predictions, and what decides them:
1. **The pose rule's truth table** (§5.5 test 1). The answer is its own state for x mod N, 2x + 1 and 2^x, and not
   for the polynomial targets or Fibonacci, in all three representations. Run before this file: it held, with 0
   disagreements (`logs/tiny_table.txt`).
2. **Unary cannot meet the budget.** No unary run comes, in either arm. High confidence.
3. **Base 2 comes first.** No target comes in decimal and not in binary under the same rule and arm. When both come,
   binary comes no later. Medium.
4. **Extrapolation is free at small N.** Every program with a digit loop that is right on the target's rows also
   passes at big x. Programs right on the rows that fail at big x are loops over units, failing on steps. Medium-high.
5. **The hidden state, found by the score.** For x² mod N, round 0's subgoal is x mod N on the rows
   (`law/mod(X,N)/rows`), in every representation and under both rules. Not for 2^x. Medium-high.

The runs, under both rules unless split:

| target | unary, src / ctl | binary, src | binary, ctl | decimal, src | decimal, ctl |
|---|---|---|---|---|---|
| x mod N | no / no (high) | round 1–2 (medium) | no (medium) | no (low) | no (medium) |
| x² mod N | no / no (high) | no; if it comes, round 3–4 (low) | no (medium) | no (medium) | no (medium) |
| 2^x mod N | no / no (high) | no (low) | no (medium) | no (high) | no (high) |

Why:
- **x mod N, binary, with the source.** The source poses the step from round 0, since the target is its own state.
  Once "subtract N" is a routine (round 0 in every held-out run), the step is learnable at cost 5:
  `L.A[INC.X] L.A[INC.X] T.mod`. Then the target is `D.2[step]`, 2 tokens.
- **Decimal.** The step is "×10, then reduce fully", about 8 tokens with the routines the learner has, beyond its
  probes (6) unless merges shorten it.
- **Unary.** The step is short, (r + 1) mod N, but it runs x times.
- **The controls.** No other source poses a step with two inputs, a state and a digit. A loop body is at most 2
  items, so the digit loop has no body that computes the step unless merges build one.
- **x² mod N.** It needs x mod N on the rows practised, then its Horner step (posed from that routine's values), then
  a square mod N routine, before `D.2[step] MOV.X.A CLR.A square` (5 tokens).
- **2^x, binary.** Its step, t ← t²·2^d mod N, needs a square and a doubling mod N, about 6–7 tokens with routines.

The comparison, predicted: the step is easiest to find in unary (cost 3, round 1), then binary (cost 5, round 1),
and decimal not within the probes. It extrapolates only from binary and decimal. So log space is what makes the step
extrapolate, at the price of a longer step, and base 2 pays the least.

What is reported for each run: the round, or "no, 5 rounds"; the program, checked at big x; where each routine came
from; whether and when the step was posed, learnable and practised; programs right on the rows but not at big x.
