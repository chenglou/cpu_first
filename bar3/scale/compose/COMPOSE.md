# The composition test: bar 3's sources in place of stage 3's law book

<!-- PLAIN -->
## In plain words

**The question.** Stage 3 builds H1's round on its digit machine from practice problems that a law book poses. Five
laws are enough (book_min, SP_ABLATION.md §5.1). Bar 3 poses its practice from the target's own rows instead: the
target on part of its rows, parts of its answer, its near misses' values, and the rest of the way from them. Can bar
3's sources replace the law book on stage 3's own machine, climb and choosing rules?

Each arm ran stage 3's tiny core as it is, on one B200, with predictions written down first (`PREREG.md`, and two
addenda for the two arms added on what the first runs showed). Every level, every arm faced the same two searches:
- the small target, x² mod N with N of 2–3 digits;
- the round at H1's own size, against 400 answer rows.

**What came: only the law book got the round.**

| arm | sources | what it practised | the round |
|---|---|---|---|
| C1 book_min (control) | the 5 laws | the six pieces, one a level | **yes**, level 5 (697 s): stage 3's own round, right on every rung of both of H1's ladders (T = 1 to 64, 768/768 each) |
| A1 | bar 3's | 23 near misses' values: loopless, 2–3 tokens, none a function of x and N | no (nothing learnable from level 6) |
| A1lp | bar 3's, hardest first | the same 23 | no (the 10-level cap) |
| A1c | bar 3's, clean values only | one value (10 x) | no (nothing learnable from level 1) |
| A2 | book_min less its reduction, and bar 3's | the five other pieces at levels 0–4, beside 23 values; no reduction | no (the cap) |
| A2c | A2 with clean values only | the five other pieces, and 10 x; no reduction | no (the cap) |
| C2 (control) | none | nothing | no |

No arm found the small target, the law book's included: its own search reaches cost 5, and the round costs 6 over the
routines. The round's search at full size found C1's round by calls. In the five arms with bar 3's sources, not one
restriction, projection or rest of the way was ever learnable. Every problem they learned was a near miss's value or
a law of the book.

**Why bar 3's sources pose nothing the round needs.**
- **The round is six pieces, each a different function:** the comparison, the link ("subtract N if you can"), the
  addition, the multiply-add row, the multiply and the reduction. The law book poses each from facts and identities
  of the data's numbers.
- **Bar 3's sources only re-describe the target.**
  - A restriction is the target on fewer rows. A projection needs the target's answer. A near miss's value is what
    the search already reaches.
  - Two routes could reach a piece: the rows where x² < N (there the target is x², the multiply's specification), and
    the rest of the way from a near miss holding x² (the reduction).
  - **The first is never posed.** On the target's own 8-probe draw at width 2, x'² < N' holds on none of the probes.
    The source poses a predicate only when it holds on 10–80% of them.
  - **The second is closed twice** (A2, A2c and §5.5):
    - x² is never a near miss: it fits 3 of the 16 probes, against 6–10 for the best 12;
    - handed the rest of the way from it anyway, the climb's searches do not find the cost-5 reduction that solves
      it. Every program they test is right on all 16 probes and wrong at width 3.
- **The near misses' values here are not functions of x and N.** The digit machine starts dirty, and the ported source
  poses whatever a near miss's prefix leaves in a register. Of A1's 23 routines, not one is clean. Bar 3's own
  machine starts clean. With clean values only (A1c), one problem was learnable (10 x), and then nothing.
- **The one-more-digit source could not have helped** (§5.3). It poses a piece only from a routine whose value is x mod
  N. Of all seven arms' routines, the only one is the book's own reduction.

**What it means.**
- On stage 3's machine, bar 3's sources replace neither the law book nor even its last law. Every piece of the round
  that any arm practised came from the book.
- What the book supplies is the specification of functions other than the target, from facts and identities of the
  data's numbers. A source that re-describes the target poses none of them.
- This is SELF_PRACTICE.md §3.3's finding (bar 3's sources "pose nothing stage 3 needs here"), now tested like for
  like:
  - the smallest book as the control, in the same harness;
  - the round's own search at H1's size in every arm, and a control with no practice;
  - two choosing rules and clean values;
  - and the book less its last law, the one place a target-derived source had a route.

**What is left** (§8):
- **The committed oracle arm's value source** poses values that read the dirty start. `clean.py`'s check is the fix,
  here behind a flag.
- **The route from x²** would need two changes to stage 3's own machinery, not to bar 3's sources:
  - near misses ranked by something other than fit;
  - a search that tests more than one spelling of a state.
- **Not run:** the generic-instruction tiny core (`integration/generic_ops/core_g.py`), and calls in the small target's
  own search.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md (frozen 06:19 PDT, sha256 `7b698b48f00a35c7b2edb1d2284ac20d9760287d58174e630b82307db8260829`),
PREREG_ADDENDUM.md (06:40, `685028d421a05fd379c71ed3147ba8138485a037a5438033ddb9e845da30162b`) and
PREREG_ADDENDUM2.md (06:46, `ae0f2f9e9ac0ecbea416dbf5b1e72613c897478075d82f6b9a7a9a8c803ba16d`), each frozen
(read-only) before the arms it names were launched, unchanged:

<!-- PREREG -->
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

# The composition test: one more arm, pre-registered before its launch

Written 2026-10-26 06:40 PDT. The five arms of PREREG.md (sha256 7b698b48…) were launched at 06:29 and were running;
this arm was not launched. COMPOSE.md §1 repeats this text unchanged.

## Why

The offline one-more-digit test (`digit_test.py`) ran on `cp_bar3`'s record as it stood at level 2. Every one of its
12 practised routines reads a register the target's inputs leave dirty, in every placement. Not one is a function of
x and N:
- the ported value source (integration/self_practice/oracle.py) poses whatever a near miss's prefix leaves in a
  register, whatever it read (the carry, A, S);
- its acceptor recomputes that value on fresh rows with the same prefix, which then always matches.

Bar 3's own machine starts clean (library/inventing/brm.py: A, T and U are 0). There every near miss's value is a
function of the inputs. So the five arms test the sources as ported. This arm tests them closer to bar 3's own.

## The change (`clean.py`, flag CLEAN_VALUES)

- A near miss's value is posed only when it is the same under a second dirty start of the target's root (the same x'
  and N'):
  - as a value problem;
  - and as the rest of the way from it.
- The projections and the restrictions are the target's own functions, unchanged.

With the flag off the driver copy runs as the five arms ran it (the diff against the launched copy,
`logs/core_compose_cp1.py.txt`, is the flag's lines only). A tiny run with the flag off gave the earlier tiny run's
records again.

## The arm

`cp_bar3_clean`: A1 (`cp_bar3`) with CLEAN_VALUES=1. Everything else is the same: one B200, RULE=lpca, NP 8, the deep
probes over whole concepts, LEVELS 10, SECONDS 2,400, the small target and the round's search each level.

## Predictions

| prediction | confidence |
|---|---|
| each level poses fewer value problems than `cp_bar3` at the same level | high |
| its practised routines are functions of x and N in the target's placement (the offline test finds them clean) | medium-high |
| none of the six pieces is practised | high |
| no small target, no round | high |
| the one-more-digit test: some of its routines are states, and none is a piece of the round | medium |

# The composition test: a seventh arm, pre-registered before its launch

Written 2026-10-26 06:46 PDT. PREREG.md's five arms and PREREG_ADDENDUM.md's `cp_bar3_clean` were launched. All but
`cp_bar3_minus_red` (A2) had ended; A2 was at level 6 of 10. This arm was not launched. COMPOSE.md §1 repeats this
text unchanged.

## Why

A2 is book_min without the reduction's pattern, plus bar 3's sources. It practised the comparison, the link, the
addition, the row and the multiply at levels 0–4, as book_min did, and no reduction at level 5. Its near misses after
the multiply (the looks at the poses of levels 5 and 6) are all built from the near-miss shortcut routines it
practised beside the book's pieces:
- each of the best 12 calls one, and fits 10 of the 16 probes;
- x² (`LDA.X STA.Y T.mul.x,o.XY`) is right only where x'² < N'. No near miss of the run calls the multiply;
- so the rest of the way from x², the one route to the reduction that the theory leaves open (COMPOSE.md §2.1), was
  never posed.

Those shortcut routines read registers the inputs leave dirty (`digit_test.py`: none of A1's 23 is clean).

## The arm

`cp_bar3_minus_red_clean`: A2 with CLEAN_VALUES=1 (`clean.py`). A near miss's value is posed, and the rest of the way
from it, only when it is a function of x' and N'. Everything else is A2's: one B200, RULE=lpca, NP 8, the deep probes
over whole concepts, LEVELS 10, SECONDS 2,400, the small target and the round's search each level.

## Predictions

| prediction | confidence |
|---|---|
| the comparison, the link, the addition, the row and the multiply, at levels 0–4 as in book_min | medium-high |
| after the multiply, a near miss holds x², or x² less a multiple of N, among the best 12 at some level's pose | medium |
| the rest of the way from such a value is posed, and is the reduction at the practice widths | certain, given the line above |
| that rest is learned: a reduction accepted at width 3 | low-medium |
| the small target or the round comes | low |
| if the round comes, it is right on every rung of both ladders | high, given the round |
<!-- /PREREG -->

### 2. Theory and design

#### 2.1 Theory

**What the round is made of.** Stage 3's round on the digit machine is sp2's `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N
T.red.x+on.XA`: x into Y, the product x·x into A and then X, N into A, then the reduction of X by N into A. Under it are
five more pieces, each a routine over the ones below it (book_min's climb, SP_ABLATION.md §5.1):

| piece | its law and pattern | program | the recurrence (LARGE_NUMBERS.md) |
|---|---|---|---|
| the comparison | lt on the pair | `CLC P[SBB.Y INC]` | a borrow chain from the lowest place (§2.2) |
| the link | red on the pair | `STA.S T.lt.pair.AY SEL.S` | subtract if no borrow: one step of long division |
| the addition | add on the pair | `CLC P[ADC.Y INC]` | a carry chain (§2.2) |
| the multiply-add row | a + b·c on (o, x, d) | `LDH.Y CLK P[MUL.X INC]` | multiplying by a digit: a carry below the base (§2.2) |
| the multiply | mul on (x, o) | `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]` | not finite-state (§2.3): a loop over the row |
| the reduction | red on x + o N | `STA.Y P[BRG.X B[T.red.pair.AY]]` | Horner's automaton, one quotient digit at a time (§2.2) |

**Why practice at all.** A search over programs by cost is Levin's search (LARGE_NUMBERS.md §3.1). Its cost grows
by a constant factor per token: on bar 3's machine 9–10 times, and on this one about 30 (C1's level-0 target search
held 35, 986, 27,608, 794,483 and 29.2M states at costs 1 to 5). The round is 6 tokens over the routines and 23 over
the primitives, its routines inlined: the multiply 9 and the reduction 10 of them. So it is out of reach unless the
pieces are routines, each found at its own cost of 3–5 tokens over the pieces below it. The no-practice control (C2)
shows the first half.

**What a source must supply.** For each piece, rows of the piece's own function: inputs and answers at sizes where
the program is cheap to find, with acceptance at a larger size, so that only a general program passes. The law book
supplies them from the data's numbers:
- facts: the order of x and N;
- identities: red(a + q N) = a;
- the learner's own routines: the sum, the row, the product.

None of them is derived from the target.

**What bar 3's sources can supply** (oracle.py; bar3/problems.py's on bar 3's machine). Each transforms rows the
learner already has:
- **Restrictions:** the target on the rows where t1 < t2. Only one turns it into a piece: x² < N, where the target is
  x² itself, the multiply's specification. At the practice widths (N' of 1–2 digits) x' is one digit there and x'² is
  a single digit product, one MUL. At the acceptance width (N' < 1,000) x' reaches 31, so only a two-digit multiply
  passes.
- **Projections:** the target's low digit, its high part, y = 0, y < x and y = x. Each needs the target's own value,
  so each is as hard as the target.
- **Near misses' values:** what a register holds after a top-level item of the target search's best wrong programs.
  Each is reachable by construction (its own prefix computes it), so practising one compresses what the search
  already reaches. On the digit machine, which starts dirty, it need not be a function of x and N at all (§2.4).
- **The rest of the way:** the target from X = a near miss's value. When that value is x², the rest is the reduction
  proper, a piece. When it is anything else, it is another problem as hard as the target.

So bar 3's sources can pose two pieces at most, and each only conditionally:
- the multiply through the restriction x² < N;
- the reduction through the rest from a near miss holding x², which needs the multiply first.

None poses the comparison, the link, the addition or the row. Without those, the multiply and the reduction cost 9
and 10 tokens over the primitives, past every probe (5.8M states reach cost 5). Even posed, they could not be
learned.

**The one-more-digit source** (DIGITS.md §2.3) needs a quantity whose rows the learner has and which is its own state
under x → 10x + d. x² mod N is not (LARGE_NUMBERS.md §5.2). x mod N is, with Horner's step (10 r + d) mod N, the
reduction's. So the source would pose a piece only from a routine whose value is x mod N. §5.3 tests every arm's
routines for exactly that.

**What could replace the book's last law.** Once the multiply is a routine, the target search's near misses can hold
x² (right wherever x² < N), and the rest of the way from such a near miss is the reduction. That is arm A2: book_min
without x + o N, with bar 3's sources.

#### 2.2 The harness (`make_core.py`, `climb.py`)

- **The driver** is an asserted copy of `integration/self_practice/core.py`, the tiny core's driver
  (`core_compose.py`, written by `make_core.py`; 7 replacements, each matched once):
  - PATTERNS and LAWS are split on ";", since three pattern names hold commas (x,d; x,o; o,x,d);
  - the climb is `climb.ComposeClimb`;
  - CLEAN_VALUES (§2.4).

  The climb, its rules and the oracle arm's sources are imported from the committed files as they are.
- **`ComposeClimb`** is `spcore.SelfClimb` with one change behind the flag ROUND_TOO. SelfClimb's target phase, with the
  small target, searches only the small target. With the flag, when the small target is not found and there are
  routines, the phase also runs SelfClimb's own round search (`search_round`): at full size, against the 400 answer rows
  (T = 1), by composition and, past 12 routines, by calls. The near misses that bar 3's sources read are gathered in the
  pose phase, which the flag does not touch.
- **The small target** is the oracle arm's (oracle.SmallTarget): x'² mod N' on the rows' top parts at widths 1–2,
  accepted at width 3. A program found for it is checked at full size against the 400 rows, as SelfClimb checks it.

#### 2.3 Its checks (`same.py`; `logs/eq_*`, `logs/same_*`)

- With both flags off, the copy gives the committed driver's records, apart from wall-clock times. Checked with two
  levels at tiny budgets (`logs/eq_orig.json` against `logs/eq_copy.json`). `same.py` compares history, records,
  library, rounds, ops, near misses and every unit's line, with times dropped. It tells two different runs apart
  (`tiny_c1bookmin` against `tiny_a1bar3`: they differ in six of the seven).
- After CLEAN_VALUES was added, the flag-off copy differs from the copy the first five arms ran
  (`logs/core_compose_cp1.py.txt`) only in the flag's lines, and a tiny run with it gave the earlier tiny run's
  records again (`logs/same_tiny_a1bar3_again.txt`).

#### 2.4 The clean-values flag (`clean.py`; PREREG_ADDENDUM.md)

`CleanTarget` is SmallTarget with one change. A near miss's value is posed only when it is the same under a second
dirty start of the target's root (the same x' and N'). That covers both the value problem and the rest of the way from
it. The projections and the restrictions are the target's own functions, unchanged.

Why: bar 3's machine starts clean (library/inventing/brm.py: A, T and U are 0), so there every value is a function of
x and N. The digit machine starts dirty (library/dsl.py), and the ported source poses any value. Its acceptor
recomputes the value on fresh rows with the same prefix, which then always matches.

#### 2.5 Settled after the freeze, before any arm ran

- **"The tiny core's budgets"** are those of its documented run (SELF_PRACTICE.md §7's command; sp2's settings):
  - NP 8;
  - deep probes over whole concepts (DEEP_BY bindings, DEEP_FOCUS closure);
  - the rest at core.py's defaults (probe 1M, fallback 1M, deep 5.8M, look 300k, round 4M, SEARCH_S 900).

  The smoke tests had used tiny budgets.
- **The smoke tests ran before ROUND_TOO existed** (their configs have no such key), so the tiny test after the freeze
  covered it (§3).

### 3. The tiny tests (local, under an 8 GB memory cap; `logs/smoke_*`, `logs/tiny_*`)

- **Smoke tests**, before the freeze, one level at tiny budgets:
  - the four configurations pose 0 (none), 24 on 12 roots (book_min), 30 on 17 (bar 3's) and 54 on 29 (A2's)
    candidates.
- **The tiny test**, after the freeze, two levels at tiny budgets with ROUND_TOO:
  - book_min practised the comparison, then the link, as its stage-3 climb does;
  - bar 3's sources practised near misses' values;
  - each level ran both searches, the small target and then the round at full size.
- **The clean-values tiny test:** level 0 posed 9 candidates on 6 roots against 37 on 20 without the flag; the one
  routine practised, `LDA.X SHL.A`, is clean (§5.3's test).
- **The checks' own tests:**
  - `verify.py ladder` gives stage 3's tiny-core round (`core_lpca_v13`) 768/768 on all 14 rungs
    (`logs/ladder_v13_check.txt`);
  - and a wrong program, the square without the reduction, 2/768 at T = 1 and 0 after (`logs/ladder_v13_negative.txt`).

### 4. The runs (`modal_compose.py`; `records/cp_*.json` and `.log`)

Each arm on one B200, from the same driver and settings (§2.5), RULE lpca unless named, LEVELS 10, SECONDS 2,400,
TARGET small, ROUND_TOO 1:

| arm | tag | SOURCES | LAWS / PATTERNS | other |
|---|---|---|---|---|
| C1 | `cp_book_min` | law | lt;red;add;mul;acc.mul / pair;x+on;x,o;o,x,d | |
| A1 | `cp_bar3` | restrict,project,value,rest | | |
| A1lp | `cp_bar3_lp` | restrict,project,value,rest | | RULE lp |
| A1c | `cp_bar3_clean` | restrict,project,value,rest | | CLEAN_VALUES 1 |
| A2 | `cp_bar3_minus_red` | law,restrict,project,value,rest | lt;red;add;mul;acc.mul / pair;x,o;o,x,d | |
| A2c | `cp_bar3_minus_red_clean` | law,restrict,project,value,rest | lt;red;add;mul;acc.mul / pair;x,o;o,x,d | CLEAN_VALUES 1 |
| C2 | `cp_none` | none | | LEVELS 1 |

Launched at 06:29 (the first five), 06:40 (A1c) and 06:46 PDT (A2c), and the route checks at 07:10 and 07:14, each
after `spend.py 4750` (LAUNCH OK; $3,596.77 billed this month at the first, $3,605.94 at the last). One spend watcher
covered the study's apps, restarted with each new one (stop each at $36, all at $36; `logs/watch_cp*.txt`).

**How they ran.**
- All seven ended on their own with exit 0. None reached its 2,400 s deadline:
  - C1 when the round came;
  - A1 and A1c with nothing learnable;
  - A1lp, A2 and A2c at the 10-level cap;
  - C2 after its one level.
- GPU memory peaked at 12–28 GiB.
- The two route checks (§5.5) ran later on the same image, one B200 each, from the Volume's final A2c record:
  `route`, 07:10, and `route_focus`, 07:14.
- **One slip in the tooling.** At 06:51, fetching the Volume's whole `runs/` folder while A2 was saving its record
  truncated the local copies to 0 bytes. Fetching again restored them, and later fetches went file by file. The
  Volume was untouched.

### 5. Outcomes (`verify.py`; `logs/verify_cp1.txt`)

#### 5.1 Each arm's climb

**C1, book_min (the control): stage 3's climb, one piece a level, then the round.**

| level | pool | practised (cost, program) | piece |
|---|---|---|---|
| 0 | 24 | lt on the pair, in A against X, Y, N (4, `CLC P[SBB.X INC]`) | the comparison |
| 1 | 12 | red on the pair, the same three (3, `STA.S T.lt.pair.AX SEL.S`) | the link |
| 2 | 12 | add on the pair, the same three (4, `CLC P[ADC.X INC]`) | the addition |
| 3 | 48 | a + b c on (o, x, d), four placements (5, `LDH.Y CLK P[MUL.X INC]`) | the multiply-add row |
| 4 | 24 | mul on (x, o), in X, Y both ways (5, `CLA W[T.acc.mul.o,x,d.AYX INC SHL.Y]`) | the multiply |
| 5 | 12 | red on x + o N, in X against A (5, `STA.Y P[BRG.X B[T.red.pair.AY]]`) | the reduction |
| round | | `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA`, and the same with `T.mul.x,o.YX` | |

The round came at level 5, 697 s, by the round's search by calls (cost 2 calls; 1, 331 and 80,782 states), after the
flat search exhausted cost 5. This is SP_ABLATION.md §5.1's climb and sp2's round, level for level. The small target's
search never found x'² mod N' (best: 12 of 16 probes at cost 5; 4M states, cost 5 streamed).

**A1, bar 3's sources: 23 near misses' values, then nothing learnable.**
- Pools of 44, 54, 52, 64, 36, 38, 34 and 34 candidates at levels 0–7.
- **Practised:** 4, 4, 4, 4, 4 and 3 problems at levels 0–5, every one a near miss's value: loopless programs of 2–3
  tokens, each built on earlier ones (`SHR.X CLA ADC.X`, then `MUL.N ADC.Y T.val.3.tgt.XN`, then
  `MUL.N CLK T.val.24.tgt.XN`).
- At level 6, no new concept was learnable; at level 7, nothing was learnable, and the climb ended (543 s).
- **Never learnable:** every restriction, every projection, every rest of the way. The 55 problems the deep probes
  chose as closest were rests and restrictions, at closeness 0.22–0.43.
- The small target's best: 10 of 16 probes. The round's search: nothing at any level.

**A1lp, hardest first:** the same 23 picks through level 5, then no new concept learnable to the 10-level cap (704
s). No small target, no round.

**A1c, clean values only:** 6 candidates a level (A1: 34–64). One learnable, `LDA.X SHL.A` (10 x, cost 2), practised
at level 0; nothing new at level 1, nothing learnable at level 2 (61 s). No small target, no round.

**A2, book_min less the reduction's pattern, with bar 3's sources: the five other pieces, and no reduction.**
- Levels 0–4: the comparison, the link, the addition, the row and the multiply, at book_min's levels and costs, each
  beside 3 near misses' values (lpca's four concepts a level).
- Levels 5–6: values only (4 each). Levels 7 and 9: nothing new. Level 8: other bindings of the row.
- The cap at level 10 (1,826 s): 42 routines (23 values), no reduction, no small target, no round.
- **Its near misses after the multiply** (the looks at levels 5–9): each of the best 12 calls a value routine and fits
  10 of the 16 probes. None in the whole run calls the multiply.

**A2c, A2 with clean values only: the five other pieces, and no reduction.**
- Levels 0–4: the same five pieces at the same levels, and 10 x at level 0.
- Level 5: 6 candidates, nothing new learnable.
- Levels 6 and 8: other bindings of the row, then of the comparison and the link (each through copies, e.g.
  `STA.Y LDA.X T.lt.pair.AY`). Levels 7 and 9: nothing new.
- The cap at level 10 (1,155 s): 24 routines, no reduction, no small target, no round.
- **Its near misses after the multiply** fit 5–6 of the 16 probes. One, at levels 5–6, calls the multiply:
  `SHL.X T.mul.x,o.XY ADC.X`. Its value after the multiply reads Y, which the inputs leave dirty, so it is not posed.
  x² itself is not among them (§5.5).

**C2, no practice:** no candidates. The small target over the primitives alone: nothing to cost 5 (4M states; best 7
of 16 probes), 3.6 s.

#### 5.2 The round's check (`verify.py ladder`; `logs/ladder_cp_book_min.txt`)

The stage-3 studies check the round on H1's two depth ladders, the round applied T times at full size (SP_ABLATION.md
§5.1: "T = 64 on both ladders, every rung 768/768"). The same, with the machine's exact cells:

| program | depth_t (seen moduli), T = 1 … 64 | depth_ood_n_t (unseen moduli), T = 1 … 64 |
|---|---|---|
| `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA` | 768/768 on all 7 rungs | 768/768 on all 7 rungs |
| `LDA.X STA.Y T.mul.x,o.YX STA.X LDA.N T.red.x+on.XA` | 768/768 on all 7 rungs | 768/768 on all 7 rungs |

Neither ladder shares an (x, N) pair or an instance with the 101,999 training rows the climb drew from, and no modulus
of depth_ood_n_t is among the training rows' 430 (checked: 0 of 5,376 rows each).

#### 5.3 The one-more-digit test (`digit_test.py`; `logs/digits_*.txt`)

What the digits study's source would pose from each practised routine, had it been ported (DIGITS.md §2.3):
- each routine's value in A on the target's inputs, in the target's placement (X = x, N = N) and in its own binding
  with x and N in both orders;
- a placement whose value changes with the dirty start (two seeds) is not a function of x and N, so nothing can be
  posed from it;
- the rows (v(x), d, N) → v(10 x + d), on x < 60 at N ≤ 60 and x < 30 at N ≤ 30.

| arm | routines | placements | clean | states | a state only in a pair with x² mod N | a piece of the round |
|---|---|---|---|---|---|---|
| C1 book_min | 16 | 40 | 24 | 17 | 0 | 1: the reduction, X = x and A = N (x mod N) |
| A1 | 23 | 46 | 0 | 0 | 0 | 0 |
| A1lp | 23 | 46 | 0 | 0 | 0 | 0 |
| A1c | 1 | 2 | 2 | 2: 10 x, and 10 N with x and N swapped | 0 | 0 |
| A2 | 42 | 87 | 22 | 16 | 0 | 0 |
| A2c | 24 | 55 | 33 | 23 | 0 | 0 |

The states other than x mod N are one-to-one in x (the comparison's difference, the sum, the product), or do not
depend on x at all, so their steps are not pieces. The positive control holds: the book's reduction, placed as the
round uses it, is x mod N, a state, and a piece. No routine of A1 or A1lp is even clean.

#### 5.4 What bar 3's sources posed (`route.py` on a B200; `records/route.log`)

The oracle arm draws its probes with a generator on the device, so only a run on the arms' GPU sees their draws.
`route.py` does, and recomputes A2c's logged near misses' fits exactly (6 of 16, four of four). `restrictions.py`, run
on a CPU, draws other probes, and its report (`logs/restrictions.txt`: four restrictions) does not describe the runs.

- **Restrictions.** The source takes every predicate t1 < t2 over 11 terms whose share of the target's last draw is
  10–80%, one per pattern, the first in order. On the arms' draws it poses two, the same every level and in every
  arm: N < 2x and 2x < N. On each the target is x² mod N on part of the rows, as hard as the target.
  - **x² < N, the multiply's specification, is never posed.** On the target's last draw (width 2), x'² < N' holds on
    0 of its 8 probes (on 3 of 8 at width 1).
- **Projections:** the target's low digit, its high part, y = 0, y < x and y = x. Never learnable: each needs the
  target's value.
- **Values:** 23 practised in A1 and A2, 1 in A1c and A2c. In A1 none is a function of x and N (§5.3).
- **Rests:** one from each value, never learnable.

#### 5.5 The open route: the rest of the way from x² (`route.py`; `records/route.log`, `records/route_focus.log`)

On the arms' GPU, with A2c's final library:
- **x² is never a near miss.** `LDA.X STA.Y T.mul.x,o.XY` (and the YX binding) fits 3 of the 16 probes. The best 12
  near misses fit 6. So the rest of the way from x² was never posed.
- **Handed that rest anyway** (X = x'², N = N'):

| program | cost | right on (of 16) | accepted at width 3 |
|---|---|---|---|
| `CLA P[BRG.X B[T.red.pair.AN]]` | 5 | 16 | yes |
| `LDA.N P[BRG.X B[T.red.pair.AN]]` | 5 | 16 | yes |
| `LDA.N STA.Y P[BRG.X B[T.red.pair.AY]]` | 6 | 16 | yes |
| `STA.Y P[BRG.X B[T.red.pair.AY]]` (the book's, which expects A = N) | 5 | 6 | no |
| `P[BRG.X B[T.red.pair.AN]]` | 4 | 6 | no |

- **The searches the climb would run on it,** each at the deep probe's 5.8M states:

| vocabulary | states by cost (0 to 5) | best fit by cost | accepted |
|---|---|---|---|
| the composition vocabulary (all of A2c's routines) | 1, 56, 2,658, 121,948, 5.6M, 309.6M streamed | 0, 7, 8, 11, 16, 16 | none (101 s) |
| the primitives, T.red.pair.AN and T.lt.pair.AN | 1, 34, 913, 24,411, 676,545, 24.3M streamed | 0, 7, 8, 11, 16, 16 | none (10 s) |
| the primitives, T.red.pair.AY and T.lt.pair.AY | 1, 34, 930, 25,268, 709,936, 25.5M streamed | 0, 7, 7, 7, 9, 11 | none (7 s) |

The second vocabulary is the deep probe's focus at its most favourable: the routines the right answer calls. States
right on all 16 probes exist at costs 4 and 5. Not one tested program passed acceptance at width 3, although the
hand-written cost-5 reduction, in the same vocabulary, does. `bfs_roots` tests each goal state in one spelling
(fewest top-level items, then fewest primitives). A look-alike spelling, wrong at width 3, is the likely reason; this
was not checked further.

### 6. Predictions against outcomes

| arm | prediction | confidence | outcome |
|---|---|---|---|
| C1 book_min | the six pieces, one concept a level, and the small target or the round by level 6 | medium-high | **right**: the six at levels 0–5, the round at level 5 (the round, not the small target) |
| A1 bar3 | only near misses' values and shortcuts; none of the six pieces; no small target, no round | high | **right** |
| A1lp | the same | high | **right** |
| A2 | lt, the link, add, the row and the multiply as in book_min | medium-high | **right**: at levels 0–4, each beside near misses' values |
| | then a near miss holding x² gives the rest of the way, a reduction, practised | low | **no**: x² never a near miss (§5.5) |
| | and the small target or the round comes | low | **no** |
| C2 none | no small target, no round | certain | **right** |
| the one-more-digit test on A1's routines | no routine whose values are a state that is a piece of the round | high | **right**: not one of them is even clean |
| A1c (addendum 1) | fewer value problems than A1 at each level | high | **right**: 6 candidates a level against 34–64 |
| | its routines are functions of x and N in the target's placement | medium-high | **right**: its one routine, 10 x |
| | none of the six pieces | high | **right** |
| | no small target, no round | high | **right** |
| | the one-more-digit test: some states, none a piece | medium | **right**: 10 x is a state (one-to-one), not a piece |
| A2c (addendum 2) | the five pieces at levels 0–4 as in book_min | medium-high | **right** |
| | after the multiply, a near miss holds x², or x² less a multiple of N, among the best 12 | medium | **wrong**: x² fits 3 of 16 probes and the best 12 fit 5–6; the one that calls the multiply reads the dirty Y |
| | the rest of the way from such a value is posed | certain, given the line above | not reached |
| | that rest is learned | low-medium | **no**: not posed; and handed over, not found by the climb's searches (§5.5) |
| | the small target or the round comes | low | **no** |
| | if the round comes, it is right on every rung of both ladders | high, given the round | not reached |

### 7. Dead ends, and how hard each was tried

| what | how hard | where it stopped |
|---|---|---|
| bar 3's sources alone (A1) | 8 levels, 543 s on a B200; 4 picks a level at levels 0–5 (23 routines); every level the probes (1M states), the deep probes (5.8M, the closest concepts whole), the small target's search (4M) and the round's at full size | nothing learnable from level 6; no piece posed |
| the same, hardest first (A1lp) | 10 levels (the cap), 704 s | the same 23 routines; nothing new from level 6 |
| the same, clean values only (A1c) | 3 levels, 61 s | one routine (10 x), then nothing learnable |
| book_min less the reduction's pattern, with bar 3's sources (A2) | 10 levels (the cap), 1,826 s; 42 routines | the five other pieces at levels 0–4; no reduction: x² never among the near misses, which all fit 10 of 16 through value routines |
| the same with clean values (A2c) | 10 levels (the cap), 1,155 s; 24 routines | the five other pieces; no reduction; other bindings of the row, the comparison and the link |
| the rest of the way from x², handed over (`route.py`) | the hand-written reductions, and three searches at the deep probe's 5.8M states: the full vocabulary, and two focused on the link | the reductions of cost 5 are right and accepted; no search accepted anything (look-alikes at costs 4 and 5) |
| the small target's own search | every level of all seven arms, 4M states: to cost 5, or to cost 4 with the bigger libraries (the last level streamed) | never found, the law book's arm included: the program costs 6 over the routines, and this search has no calls |
| the one-more-digit source, offline | every practised routine of every arm, in 2–3 placements, on two grids | only the book's own reduction is a piece |
| the restrictions rebuilt on a CPU (`restrictions.py`) | one run | torch's generators differ by device, so it drew other probes than the runs and listed four restrictions, two of which the runs never posed (1 < x, x < 2); `route.py`, on the arms' GPU, replaced it |

### 8. What is left

- **The committed oracle arm's value source** (integration/self_practice/oracle.py) poses near misses' values that
  read the dirty start. `clean.py`'s check is the fix; here it is a flag, and the committed file is unchanged.
- **The route from x² needs two changes to stage 3's machinery**, not to bar 3's sources:
  - **near misses ranked by more than fit.** x² fits 3 of 16 probes and the shortcuts 6–10. SUBGOAL.md's table view
    would see that the target is a function of (x², N). But for x² mod N itself SUBGOAL.md found the score "singles out
    nothing", and its law view is off for a target that is a law of X and N;
  - **a search that tests more than one spelling of a goal state**, or acceptance on more rows. Handed the rest, the
    searches did not reach the cost-5 reduction that solves it.
- **The restriction to x² < N** is posed only when 10–80% of the target's 8-probe draw satisfies it. Here none did.
  At width 2 about 1/√N' of the probes satisfy it (10–30%), near the floor, so more probes might pose it. Even then,
  without the row the multiply is 9 tokens over the primitives, past every probe.
- **Not run:**
  - the generic-instruction tiny core (`integration/generic_ops/core_g.py`). Nothing in the argument depends on the
    instruction set, but the dirty values and the look-alikes might;
  - calls in the small target's own search, with which the small target would come wherever the round does;
  - more seeds. Every outcome here is the same in each variant, and book_min's climb matches its 8 stage-3 seeds.
- **The positive question this leaves.** SP_ABLATION.md §5.2's generated book poses the pieces from a few rules about
  the data's numbers (the division law's form on ordered pairs, the machine's operations), and certifies on all 8
  seeds. Can those rules be found from the data rather than written?

### 9. Cost (`cost.py`; `logs/cost_cp1.txt`, `logs/watch_cp*.txt`)

| app | what | billed (13:00–14:00 UTC) |
|---|---|---|
| ap-2lba6KdE5XuYVdUqEKat2C | PREREG.md's five arms | $7.45 |
| ap-ootToK0KizdVnVoR8wEM5s | A1c | $0.31 |
| ap-PuOQEjmeAGlzrtZ28CxZkk | A2c | $1.41 |
| ap-dHPW3PcxAbLqiVENqDfkC5, ap-VjGEPxGPHeXM9e82UM90Mq | the two route checks | not yet billed |
| all | | **$9.17** billed |

The hour from 14:00 UTC was not yet billed when this was written. In it ran about 15 B200-minutes: A2c's last 8, the
two route checks, and A2's last half minute. That is about $1.6 at the first hour's rate, so **about $11 in all**.
The watcher's last estimate (14:29 UTC), $13.75, is an upper bound: it charges the whole hour under way at the last
complete hour's rate.

Soft budget $30, cap $40: under both. One spend watcher covered every app (stop each at $36, all at $36), and
`spend.py 4750` passed before each of the five launches ($3,596.77 to $3,605.94 billed this month). All five apps are
stopped. Local work, under an 8 GB memory cap:
- the smoke and tiny tests, the equivalence checks and the offline one-more-digit tests;
- the ladder checks (seconds each);
- the CPU route smoke test (about 9 minutes).

### 10. Files and how to rerun

- `PREREG.md`, `PREREG_ADDENDUM.md`, `PREREG_ADDENDUM2.md`: the pre-registrations, read-only, hashes in §1.
- `make_core.py` writes `core_compose.py`, the driver: integration/self_practice/core.py with 7 asserted
  replacements. `climb.py`: ComposeClimb (ROUND_TOO). `clean.py`: CleanTarget (CLEAN_VALUES).
- `modal_compose.py`: the seven arms (ARMS) and the two route checks (CHECKS), one B200 each, app and Volume
  `bar3-compose`.
- `verify.py` (`summary`: each arm's levels; `ladder`: a program on H1's two depth ladders), `digit_test.py` (§5.3),
  `route.py` (§5.4–5.5, on the arms' GPU), `restrictions.py` (the CPU rebuild that does not describe the runs, §7),
  `same.py` (two records the same apart from times), `cost.py` (§9).
- `records/cp_*.json`, `.log`: the arms' records and logs; `records/route.log`, `route_focus.log`: the route checks;
  all fetched from the Volume.
- `logs/`: the smoke and tiny tests (`smoke_*`, `tiny_*`), the copy's checks (`eq_*`, `same_*`,
  `core_compose_cp1.py.txt`), the launches (`launch_*.txt`), the watcher (`watch_cp*.txt`), the outcomes
  (`verify_cp1.txt`, `ladder_*.txt`, `digits_*`, `restrictions.txt`, `route_cpu_smoke.txt`, `cost_cp1.txt`).

```
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/compose/make_core.py
cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose/modal_compose.py --arms cp_book_min,cp_bar3,cp_bar3_lp,cp_bar3_minus_red,cp_none
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose/modal_compose.py --arms cp_bar3_clean,cp_bar3_minus_red_clean
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose/modal_compose.py --arms route,route_focus
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/compose/modal_compose.py --mode fetch
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.verify summary cp_book_min cp_bar3 cp_bar3_lp cp_bar3_clean cp_bar3_minus_red cp_bar3_minus_red_clean cp_none
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.verify ladder cp_book_min "LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA"
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.digit_test bar3/scale/compose/records/cp_bar3.json
cd cpu_first && H1_DATA=~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2 ~/github/sotaku/venv/bin/python -m bar3.scale.compose.restrictions
```

A tiny test locally (CPU, about a minute, under a memory cap), as in §3:

```
cd cpu_first && bash -c 'export H1_DATA=~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2; env TARGET=small ROUND_TOO=1 RULE=lpca NP=8 DEEP_BY=bindings DEEP_FOCUS=closure PROBE=20000 FALLBACK=20000 DEEP=50000 LOOK=10000 ROUND=20000 SEARCH_S=60 LEVELS=2 OUT=bar3/scale/compose/logs SOURCES=law "LAWS=lt;red;add;mul;acc.mul" "PATTERNS=pair;x+on;x,o;o,x,d" TAG=tiny_c1bookmin bar3/scale/rowcheck/memcap.sh 8 ~/github/sotaku/venv/bin/python -W ignore -m bar3.scale.compose.core_compose'
```
