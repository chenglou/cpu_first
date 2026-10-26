# The composition test: pre-registration

Written 2026-10-26 06:19 PDT, before any run of an arm (only the smoke tests below ran, at tiny budgets). COMPOSE.md §1
repeats this text unchanged.

## The question

Can bar 3's own sources replace stage 3's law book on stage 3's digit machine? Stage 3's climb builds H1's round from
practice problems a law book poses. A 5-law book is enough (book_min: lt and red on the pair, add on the pair, the
multiply-add row a + b c on (o, x, d), the multiply on (x, o), the reduction on x + o N; SP_ABLATION.md §5.1). Bar 3's
sources pose practice problems from the target's own rows: its restrictions, projections, its near misses' values,
and the rest of the way from them (ported to the digit machine as the oracle arm, integration/self_practice/oracle.py).

## Theory, in brief

- **Every piece of the round is a digit-serial recurrence** (theory/LARGE_NUMBERS.md §2.2):
  - the comparison and the subtraction are borrow transducers, and the addition a carry transducer;
  - the multiply-add row is a carry chain of digit products;
  - the multiply is a recurrence over the multiplier's digits, shift and add the row;
  - the reduction is a recurrence over the dividend's digits: long division, Horner's step.

  The one-more-digit test (DIGITS.md) is the generic way to pose such steps, but only for a quantity whose rows the
  learner has.
- **The law book's role is to supply those quantities' rows.** Comparisons are facts of the data's numbers; reductions
  follow from the identity red(a + q N) = a; sums and products come from the learner's own routines. None of them is
  derived from the target.
- **Bar 3's sources only transform rows the learner already has:** the target's, and its routines' values. From x² mod
  N alone, the target is not its own state under one more digit (it needs x mod N beside it: the truth table of
  LARGE_NUMBERS.md §5.2, which holds on any machine). Its near misses at the practice widths are small-N shortcuts
  (SELF_PRACTICE.md §3.3). So nothing posed from the target is a piece of the round. Every program that computes one
  costs more than a probe reaches from the primitives: the comparison alone, `CLC P[SBB.Y INC]`, is 4 tokens.
- **The pieces may not be needed by name.** Once the multiply exists, a near miss of the target can hold x² itself
  (right wherever x² < N), and "the rest of the way" from it, x² mod N from X = x², is a reduction. That could stand in
  for the book's last law.

## Design (`make_core.py`, `climb.py`)

**The harness.** Stage 3's tiny core, as it is:
- the climb is `spcore.SelfClimb`: the digit machine, exact cells, rows' top parts at widths 1–2 for practice,
  acceptance at width 3, and the rules lp, lpca and ct2a2;
- it runs through an asserted copy of `integration/self_practice/core.py` (`core_compose.py`) with two changes:
  - PATTERNS and LAWS are split on ";", since three pattern names hold commas;
  - the climb is `climb.ComposeClimb`, which is SelfClimb unless ROUND_TOO=1.

With the flag off the copy gives the committed driver's records exactly, apart from wall-clock times (checked: two
levels at tiny budgets, `logs/eq_*`).

**The target and its checks.**
- **The target** is the oracle arm's small target: x'² mod N' at the practice widths (N' of 1–2 digits), accepted at
  width 3 (N' of 3 digits). That is x² mod N with N of 2–3 digits.
- **The round's search too.** With ROUND_TOO=1, each level where the small target is not found, the round is also
  searched as the law-book arm searches it: at H1's full size against 400 answer rows (T = 1), by composition and,
  past 12 routines, by calls. So every arm faces the same two searches.
- **The checks.** A small-target program is checked at full size against those 400 rows. A round must reproduce all
  400.

## Arms

Each on one B200 (Modal, this folder's `modal_compose.py`), the tiny core's budgets, seed 0, RULE=lpca (the concept
rule that found the round in SELF_PRACTICE.md §3.4), LEVELS 10, SECONDS 2,400:

| arm | sources | law book |
|---|---|---|
| C1 `book_min` (control) | the law book | lt, red, add, mul, acc.mul on pair, x+on, x,o, o,x,d |
| A1 `bar3` | the target's restrictions, projections, near misses' values and rests | none |
| A1lp `bar3_lp` | the same, RULE=lp (bar 3's hardest first) | none |
| A2 `bar3_minus_red` | the law book and the target's sources | book_min without the x+on pattern: no reduction law |
| C2 `none` (control) | none: no practice | none (LEVELS 1) |

Not ported, and why:
- **The one-more-digit source**, as a poser in the climb. On this target it poses nothing from the target, and it
  needs a routine whose values are a state. Instead, every arm's routines are tested offline for that property
  (COMPOSE.md §5): a state found there would have been posed.
- **Bar 3's choosing aids** (the subgoal score, the duplicate check on rows, the tie-break). They choose among what
  the sources pose, so they cannot help an arm whose sources pose nothing the round needs.

Money: soft $30, cap $40. Every launch is guarded by `spend.py 4750`, with one spend watcher for the app; local runs
under `memcap.sh`.

## Predictions

| arm | prediction | confidence |
|---|---|---|
| C1 book_min | the six pieces, one concept a level (lt, link, add, the row, the multiply, the reduction), and the small target or the round by level 6 | medium-high |
| A1 bar3 | practises only the target's near-miss values and shortcuts; none of the six pieces; no small target, no round | high |
| A1lp | the same | high |
| A2 bar3 minus red | lt, link, add, the row and the multiply as in book_min | medium-high |
| | then a near miss holding x² gives "the rest of the way", a reduction, practised | low |
| | and the small target or the round comes | low |
| C2 none | no small target, no round | certain |
| the one-more-digit test on A1's routines | no routine whose values are a state that is a piece of the round | high |

What is reported for each arm: its levels, its practised routines and which of the six pieces each is, whether and
where the small target or the round came, the programs and their checks at full size, its seconds and cost.
