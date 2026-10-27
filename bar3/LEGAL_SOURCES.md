# Bar 3's practice sources inside a training run: an audit

*Written 2026-10-26, while composition round 4 (`bar3/scale/slice/`) ran. Nothing was spent: no Modal, three local
checks on H1's training split (§2, §4.2, §7). Each claim cites the code or write-up it rests on. Costs marked as
estimates were not measured.*

<!-- PLAIN -->
## In plain words

**The question.** Bar 3's learner poses its own practice problems from many sources: a law book; the target's own rows,
restricted, projected, its near misses and the rest of the way from them; recurrences; constants; one more digit; and
now the target's laws cut to one digit place. Offline, every answer they give is computed with Python's or numpy's
arithmetic. On stage 3's digit machine these sources have begun replacing the law book's laws. To count, they must be
rebuilt inside a legal training run, where no training target may be computed that way (rule 14;
[LEGAL_PLAN.md](../LEGAL_PLAN.md) §1; [SELF_PRACTICE.md](../integration/SELF_PRACTICE.md) §2.2). A target may come from
the data's rows, from facts and identities, or from the learner's own routines. Which sources can move as they are,
which need a redesign, and what would the redesign cost?

**What decides most verdicts: in a run, the target's table does not exist.**
- Offline, bar 3 computes the target on every pair x < N for small N. The composition harness computes x'² mod N' on
  the rows' top parts, at 1–2 digits. Both are Python's arithmetic.
- In a run, the target's only answers are H1's rows at full size (13- to 20-bit N). The rows that show one squaring
  are the 1,000 with T = 1. A source that reads the target is legal only if it reads those.
- Counted on H1's training split (§2): at T = 1 there are 2 pairs of rows one apart (x and x + 1, the same N), no three
  in a row, no x = 0 or x = 1, and no row with all its "one more digit" children. Grouping the T = 1 rows by
  (x² mod N, N) saves 5 rows of 1,000. At T = 4, 8 and 16 there are about 2,100 pairs one apart each, but there the
  target is x^16 mod N and beyond.

**The verdicts** (§3 has each source's code, answers, redesign, cost and test).

| source (§) | answers now | in a run as it is | legal version; its cost |
|---|---|---|---|
| the law book (3.2) | Python | no | stage 3's book, as sp2 runs it; no cost |
| restrictions (3.3) | the target's; rows picked by Python | no | predicates by routines; on H1 none posed: drop |
| projections (3.4) | numpy on the answers | no | digit reads and equality; optional |
| near-miss values (3.5) | the learner's routines | yes | ranked by the law view, not by fit |
| rests, routine remainders (3.6) | Python's table | no | on the T = 1 rows at full size; ~2–3x a probe state |
| recurrences, two-number step (3.7) | Python's table | no | no rows in H1; legal on routines only |
| running sum, offset (3.8) | Python's table, mod N | no | no rows in H1 |
| constants (3.9) | Python's mod | no | no rows in H1 |
| one more digit (3.10) | Python's table | no | on routines only; stage 3's red on Bx+d is one |
| row-based duplicate check (3.11) | the problem's own rows | yes | as it is |
| subgoal score, table view (3.12) | the target's rows | yes | says nothing at T = 1 |
| subgoal law view, LAWNEAR (3.12) | numpy, torch | a decision (§6) | the learner's chain and comparison; est. seconds |
| slice cuts (3.13) | Python's integers | no | the book's own constructions; likely no level |
| slice law reader (3.13) | numpy | a decision (§6) | none before a multiply exists |
| dreams, choosing rules, probes (3.14) | routines, counts | yes | as they are |

**The composition route, rebuilt legally.** Round 2 got H1's round with bar 3's sources in place of the book's last
law (B2rd), and round 4's first arm also with the target's cuts in place of the book's row (s4_x2, its study still
running). Every target-derived step of that route read the small target, by Python: LAWNEAR's law test, the subgoal
score, and the rest of the way from x². Rebuilt legally (§4):
1. **x² as a value** is held by the round's own search at full size. Its answers are the learner's multiply's: legal.
2. **The law test** ("the answer is x² mod N") runs the learner's own routines: its chain, ten conditional
   subtractions, exact below 11·N; and its comparison, which finds the rows where the chain is exact. Then it compares
   digit strings. Simulated on the harness's own 400 answer rows, x² passes on all 8 rows where the chain reduces, and
   none of eight other plausible values passes (§4.2).
3. **The rest of the way from x²** is posed on the T = 1 rows at full size: X = x² by the learner's multiply, the
   answer each row's own.
4. **The cuts** take their answers from constructions stage 3's book already uses: repeated addition for the row, the
   identity red(x + d·N) = x for Horner's step. They then wait for the addition, as the book's laws do. B2rd, whose row
   is the book's, got its round at level 5, as s4_x2 did.

**What cannot move: the slice source's law reader.** It tests the target's law with numpy. No routine of the learner can
test mod(mul(X, X), N) before a multiply exists. Without the reader, posing the cuts means posing every operation's
cuts: a book of cuts.

**What it might cost.** The small target's own search goes: 19–35 s a level in B2rd. The law test adds one more look at
the round search's cheap states on all 400 answer rows: likely seconds a level. The rest's probes run at full size, 16
places instead of 2–6: perhaps 2–3 times the cost per state. B2rd's 10M-state deep probe took 26 s. All three are
estimates. The open question is reach, not time: do the probes still find the 4- or 5-token reduction when each state
holds full-size numbers?

**The cheapest test** (§5). Offline, from s4_x2's recorded library at level 5, on one B200, with no climb:
1. the law test by the learner's routines, over the round search's cheap states on the 400 answer rows: does it pick x²?
2. the rest of the way from x², posed at full size and probed at 1M states: does it find `LDA.N P[BRG.X T.red.Bx+d.AN]`?

About $1–2 on the B200, after writing a full-size target mode for the harness.

**A question for the judge** (§6). The rule names targets. Bar 3 also decides with Python's arithmetic: the law view,
the reader, the restrictions' predicates, the constants. They choose what to practise, and compute no target. The
entry, fast2, goes further: its miner computes other relations of the read numbers, but only order and nonzero facts
feed its lessons (REVIEW.md §5, rule 14). If decisions count too, the law view has a routine version and the reader has
none.

**What does not change.** Recurrences, the two-number step, the running sum, the offset, the constants and one more
digit stay offline sources, for bar 3's breadth: Fibonacci, the triangular numbers, 10·x² + 1. H1's rows cannot pose
them, and H1 does not need them (LEGAL_PLAN.md §1, item 4). On the learner's own routines they are legal.
<!-- /PLAIN -->

## Details

### 1. The standard

**Three things a source makes.** The rule is about targets. A strict reader asks the same of two more things a source
makes, and the certified files already hold themselves to it:

| what | legal from | not legal from |
|---|---|---|
| a target: the answer a program must give | the data's rows; facts; identities; routines | arithmetic on values |
| an input: what a problem is posed on | the data's numbers, digits; digit moves; routines | arithmetic: made-up data |
| a decision: what to pose, keep, probe deeper | counts of right rows; search costs; facts | arithmetic on values (§6) |

- **Facts** are what the data's numbers show without computing: the order x < N (H1 draws x below N), digit-string
  equality, a number's own digits. The entry fast2's miner feeds its lessons only order and nonzero facts
  ([REVIEW.md](../REVIEW.md) §5, rule 14).
- **Identities** hold by construction: red(a + q·N) = a and div(a + q·N) = q for a < N; sub(a + N, N) = a; an addition
  checked by sub.inverse; a product as repeated additions (`spcore.law_target`,
  `integration/self_practice/spcore.py:202`).
- **Routines** are programs the learner practised, run on its own machine. A routine learned from Python's answers
  would carry them in, so in a legal run every routine must itself come from legal targets. Nothing bar 3 learned
  offline can be imported.
- **Inputs** built by the learner's addition (x' + d·N') and by moving digits (B·x' + d) are stage 3's own practice
  (`spcore.Draw.term`, spcore.py:167). Inputs summed by numpy, as bar 3's book builds x + N, are not.
- Every source that reads rows needs the run to keep them: LEGAL_PLAN.md §3, question 3.

**The reference.** Stage 3's law book (`spcore.py:44–68`, `law_target` at :202) computes no number from a value:
every target is a fact, an identity or a routine, and a law is posable only once the routines its construction calls
exist. Every legal version below uses only its moves.

### 2. What a run's rows can show (local count, $0)

H1's training split, `one-layer-deeper/data/generated/<H1>/train.jsonl`: 101,999 rows. T = 1 and T = 2 have 1,000
rows each; T = 4, 8 and 16 have 33,333 each. The training moduli are 11 of 13 bits, 42 of 16 bits and 377 of 20 bits.
x is a unit, drawn from 1 to N − 1 (`data/squaring_mod.py`, `_sample_unit`). The T = 1 rows spread over 272 moduli,
3.7 rows a modulus.

| what the rows hold (same N and T) | T = 1 | T = 4 (8, 16 alike) | needed by |
|---|---|---|---|
| x and x + 1 | 2 | 2,109 | recurrences, running sum |
| x, x + 1 and x + 2 | 0 | 392 | order 2, the two-number step |
| x and at least one child: 2x + d; 10x + d | 3; 2 | 2,007; 1,068 | (one more digit needs every child) |
| x and every child: 2x and 2x + 1; all ten 10x + d | 0; 0 | 193; 0 | one more digit, base 2; base 10 |
| x = 0, x = 1 | 0, 0 | 0, 1 | offset, constants |
| x² < N | 5 (harness's 400: 2) | 223 | x² < N restriction; the identity law test |
| x² < 11·N | 20 (harness's 400: 8) | — | the chain's law test (§4.2) |
| rows saved grouping by (x² mod N, N) | 5 of 1,000 | 2,990 of 33,333 | the subgoal's table view |

The same (N, x) appears at T = 1 and T = 2 in 4 rows, at T = 2 and 4 in 61, at T = 4 and 8 in 2,279. "The harness's
400" are the composition harness's own answer rows (`compose3/core_c3.load_rows(400, seed=1, T=1)`).

### 3. Source by source

Each source: its code; where its answers come from (the data, identities, the learner's routines, or Python's
arithmetic); whether it runs in a training run as it is; a legal version and its cost on the GPU; the cheapest test.

#### 3.1 The target, and what accepts a program

- **Code.** Bar 3: `problems.target_gen` (`bar3/problems.py:197`), the target on every pair x < N up to the size (N ≤ 7
  to search, N ≤ 10 to check; later studies accept to N = 40 or 120, and on big x). The composition harness
  (`scale/compose3/core_c3.py`): the round, against 400 of H1's T = 1 rows read as digit strings; and the small target,
  x'² mod N' on the rows' top parts (`integration/self_practice/oracle.py:57`), other formulas by TARGET_FN
  (core_c3.py:56).
- **Answers.** Python's arithmetic for bar 3's target and the small target; the data's rows for the round.
- **As it is.** The round and its acceptance: yes, as sp2 runs them. The small target: no.
- **Legal version.** The T = 1 rows at full size are the target, as in sp2. Every source below that reads "the target's
  rows" must read those. The family's other formulas have no rows in H1: they stay offline.

#### 3.2 The law book's posing

- **Code.** `problems.laws` (problems.py:121): 12 operations and succ, pred on X = x, x + N, x + qN, x + QN.
  `problems.laws2` (:165): one operation on another's result, the target mod(mul(X, X), N) among them.
  `digits/source.row_laws` (`scale/digits/source.py:131`): the book on the target's own rows.
- **Answers.** Python's operations (problems.py:97). The inputs x + N, x + qN and x + QN are numpy sums too.
- **As it is.** No: neither the targets nor the inputs.
- **Legal version.** Stage 3's law book, which sp2 already runs:
  - lt, eq, min and max from the order fact and digit equality; red and div from red(a + q·N) = a and its quotient q;
    sub from sub(a + N, N) = a and the pair's own cases; add by a check (red and lt); mul and a + b·c by repeated
    addition. Composite inputs are built by the learner's own addition.
  - Bar 3's other operations: monus and absdiff are sub where an identity gives them; succ is add with the digit 1.
    gcd, N mod x and N div x have no identity and no routine to give them: they go.
  - A depth-2 law needs both of its routines first. The one that matters, mod(mul(X, X), N), is the round itself.
- **Cost.** None: it is sp2's book, certified in the run.
- **Cheapest test.** None needed.

#### 3.3 Restrictions

- **Code.** `problems.restrictions` (problems.py:209): the target on its rows where t1 < t2, terms 1, 2, 3, X, N,
  X + X, N + N, X + N, X·X, X·N, N·N (:202), kept on at least 6 rows and at most 80% of them. The port in
  `oracle.SmallTarget.pose`, kept on 10–80% of a draw. Round 2's RESTRICT_ROWS measures the share on 4,096 rows' top
  parts (`compose2/hooks.py:276`).
- **Answers.** The target's own, on fewer rows. Which rows is a predicate computed with Python's arithmetic: a decision.
- **As it is.** The answers would be the data's. The predicate is legal only if decisions may use arithmetic (§6).
- **Legal version.** The predicate by the learner's routines (its multiply, addition and comparison) on the T = 1 rows,
  once those routines exist. Predicates without arithmetic (X against N, against the digits 1–3) are facts.
- **What it would pose on H1.** x² < N holds on 5 of the 1,000 T = 1 rows: too few for either filter. So the one that
  posed the square at the practice widths (COMPOSE3.md) is never posed at full size. Others, such as 2x < N on about
  half the rows, re-pose the target on fewer rows. No restriction was ever learnable in rounds 1–3, and with
  RESTRICT_ROWS off B2rd still got its round at level 5 (COMPOSE2.md).
- **Cost.** None: drop it from the legal build.
- **Cheapest test.** None needed.

#### 3.4 Projections

- **Code.** `problems.projections` (problems.py:238): y mod 2, y div 2, [y = 0], [y < x], [y = x]. The decimal port,
  `oracle.target_goal` (oracle.py:61): the low digit, y div 10, [y = 0], [y < x], [y = x].
- **Answers.** numpy and Python on the target's answers.
- **As it is.** No. But on a decimal machine four of the five need no arithmetic: the low digit and the rest are the
  answer's own digits, moved; [y = 0] and [y = x] are digit-string equality. [y < x] is an order fact between two of
  the row's numbers, as the miner finds x < N, or the learner's comparison. Bar 3's binary pair (mod 2, div 2) is
  arithmetic on a decimal machine: it goes.
- **Legal version.** Those reads, on the T = 1 rows at full size.
- **Cost.** One more root a level. Optional: no projection was ever learnable in rounds 1–3.
- **Cheapest test.** None needed.

#### 3.5 Near-miss values

- **Code.** `problems.from_near_misses` (problems.py:273): every register's value after each top-level item of a near
  miss, run on the machine. The port, `oracle.prefix_values` (oracle.py:86), with CLEAN_VALUES (`compose/clean.py`:
  the same value under a second dirty start). LAWNEAR's held states (`compose2/hooks.py:242`).
- **Answers.** The learner's own programs, run on its own machine: legal.
- **As it is.** The values and the clean check, yes. What breaks is how near misses are chosen: by fit, the share of
  the target's answers a program gets right. At full size a wrong program is right on a row only by coincidence: x²
  is right on 2 of the harness's 400 answer rows, the rows where x² < N. So fit carries no signal at full size, and the
  law view has to rank the values (§3.12), as LAWNEAR already does beside fit.
- **Legal version.** The round search's held states at full size, ranked by the law view by routines.
- **Cost.** None beyond the law test's (§4.3).
- **Cheapest test.** §5, test 2.

#### 3.6 Rests of the way and routine remainders

- **Code.** The "remainder" in `problems.from_near_misses`: the target's answer from X = a near miss's value.
  `sources.routine_remainders` (`scale/sources.py:209`): from every routine's value. The port's "rest" roots
  (`oracle.OracleRoot`, oracle.py:97).
- **Answers.** The target's, from bar 3's table or the small target: Python's arithmetic. The inputs are the learner's
  values.
- **As it is.** No.
- **Legal version.** On the T = 1 rows at full size: X is the routine's value on the row's x and N, the answer the
  row's own. Two consequences:
  - Bar 3 keeps a rest only when it is a function: equal inputs, equal answers. At full size that test is nearly
    vacuous. The 1,000 T = 1 rows hold 1,000 distinct (x, N), so only a value that collapses many rows can fail it.
  - At the practice widths, the only legal answers for a reduction come from red(x + q·N) = x, and that is the book's
    own reduction law. A rest of the way can be legal only at full size.
- **Cost.** Probes at full size: 16 places instead of 2–6, so perhaps 2–3 times the cost per state (an estimate).
  s4_x2's rest from x² was solved by the regular 1M probe at level 5; B2rd's by REST_DEEP's 10M probe, in 26 s at the
  practice widths.
- **Cheapest test.** §5, test 2.

#### 3.7 Recurrences and the two-number step

- **Code.** `sources.recurrences` (scale/sources.py:122). Order 1: y(x + 1) from y(x) and N. Order 2: y(x + 2) from
  y(x) and y(x + 1), as one number or as the step that moves the pair (Fibonacci's). Built on `sources.table` (:105):
  every x < N of each N.
- **Answers.** The target's table: Python's arithmetic offline; in a run they would be the data's rows.
- **As it is.** There are no rows to pose them:
  - At T = 1, H1 has 2 pairs one apart and no triple; a problem needs at least 4 rows.
  - At T = 4, 8 and 16 it has about 2,100 pairs and 395 triples each. But there the target is x^16 mod N or beyond,
    with no recurrence of order 1 or 2. And with almost no (y(x), N) repeated, the source's function test passes
    anyway: it would pose a problem with no law.
- **Legal version for H1.** None, and none needed. Along T the recurrence is the round, and the T = 1 rows already are
  its step from x: sp2's search. On the learner's own routines the source is legal: a routine R's step, R(x + 1) from
  R(x), with x + 1 built by the learner's addition. No study has posed that.
- **Cost, test.** None.

#### 3.8 The running sum's step and the offset

- **Code.** `newton.sources.running_sum` (`scale/newton/sources.py:31`): y(x + 1) from y(x) and x.
  `newton.sources.offset` (:56): a rest of the way's answer less its answer at X = 0, mod N (line 65).
- **Answers.** The target's table, and for the offset a subtraction mod N: Python's arithmetic.
- **As it is.** No. The running sum needs the rows one apart of §3.7. The offset needs a row where the rest's input is
  0. That input is a routine's value, and for x² mod N, or any power of x, a unit never gives 0. H1's x are units.
- **Legal version for H1.** None. The subtraction could be the learner's (sub, then the link), but the anchor row does
  not exist.
- **Cost, test.** None.

#### 3.9 Constants

- **Code.** `sources.constants` (scale/sources.py:180): the smallest c with c mod N equal to the answer at x = 0 or 1
  (line 196). `slice/lawread.constants` (`scale/slice/lawread.py:24`): the same and their difference, as the law
  reader's leaves.
- **Answers.** Python's mod over candidate c.
- **As it is.** No. H1 has no x = 0, and no x = 1 at T = 1 (where the answer would be 1 anyway). x² mod N needs no
  constant.
- **Legal version for H1.** None.

#### 3.10 One more digit

- **Code.** `digits.source.digit_steps` (scale/digits/source.py:90): is the answer at x·B + d a function of the answer
  at x, d and N? `step_rows` (:38) needs every child x·B + d, d < B, of a row among the rows. The state may be the
  target's value or a routine's. `row_laws` (:131) poses the book on the target's rows (§3.2).
- **Answers.** The target's table or a routine's values.
- **As it is.** No rows: at T = 1, H1 holds a row and one of its children twice in base 10 and three times in base 2,
  never all of them.
- **Legal version.** On routines only: a routine's values at x' and at B·x' + d, the latter built by moving digits, as
  stage 3 builds its Bx+d pattern (spcore.py:182). Stage 3's red on Bx+d, answered by the learner's chain
  (spcore.py:254–258), is already this source's legal form for the reduction.
- **Cost, test.** None for H1.

#### 3.11 The row-based duplicate check

- **Code.** `rowcheck.solves` (`scale/rowcheck/rowcheck.py:33`): a program run on a problem's own rows and compared;
  ROWCHECK in round 2.
- **Answers.** None of its own: it compares programs on a problem's rows.
- **As it is.** Yes, wherever the problem's rows are legal. Off, B2rd still got its round at level 5.
- **Cost, test.** None.

#### 3.12 The subgoal score, LAWNEAR and the tie-breaks

- **Code.** `subgoal.score.table_view` (`scale/subgoal/score.py:106`): group the target's rows by (g, N), (g, x) or g;
  the rows saved when every class wants one answer. `subgoal.score.law_view` (:168): the target = op(a, b) or
  op(op'(a, b), c) over g, X, N and the constants, by numpy. `compose2/subgoal_c2.py` ports both. `hooks._law1`
  (compose2/hooks.py:217) is LAWNEAR's depth-1 test, torch arithmetic on the GPU. TIEBREAK and TIEFOLLOW use the score.
- **Answers.** None: these are decisions. They single out a candidate (a deeper probe, ties won) and, for LAWNEAR,
  choose which held state is posed as a value.
- **As it is.**
  - The table view needs only equality: legal. On H1's T = 1 rows it says nothing: grouping by (x² mod N, N) saves 5
    rows of 1,000. On the T ≥ 4 rows, where a 13-bit modulus carries up to 1,448 rows, the same grouping saves 2,990
    of 33,333 and the target is a function of it. That is the round composed with itself, not the round.
  - The law view computes each law with Python's arithmetic: legal only if decisions may (§6).
- **Legal version.** Test each law with the learner's routine for its operation, compare digit strings, and only on the
  rows where the routine is exact, as the learner's comparison says:
  - mod(g, N) by the chain, ten conditional subtractions, `B[link]`: the book's red on x + d·N, or the slice's step.
    It is exact where g < 11·N, and only there is its result below N, so the comparison finds those rows. The link
    alone is exact where g < 2·N;
  - add and mul by the learner's addition and multiply, on every row; lt and eq by its comparison and digit equality;
  - an operation with no routine is not tested.

  Simulated on H1's rows (§4.2), the chain's test picks x², as numpy's does.
- **Cost.** The round search's cheap states rerun on the answer rows and their values put through the chain: an
  estimate, seconds a level (§4.3).
- **Cheapest test.** §4.2, done locally; §5, test 2.

#### 3.13 The slice source (read only: composition round 4 is running)

- **Code.** `slice.hooks.slice_target` (`scale/slice/hooks.py:383`): the four cuts' answers by Python's integers: the
  row o + x·d, x·d, Horner's step (B·x + d) mod N, and x + d. `slice.lawread.read` (lawread.py:43): the target as a
  left-deep chain of the law view's operations over X, N and the data's constants, by numpy, on the small target's
  rows.
- **Answers.** Python's arithmetic, both.
- **As it is.** No.
- **Legal version of the cuts.** Each is a law stage 3's book already answers without arithmetic:
  - the row, a + b·c on (o, x, d): d additions of x, then o added (`acc.mul`, spcore.py:265);
  - x·d: d additions (`mul` on (x, d));
  - Horner's step: red on x + d·N, by the identity red(x + d·N) = x, its input built by the learner's addition. It is
    the same function as red on B·x + d: both reduce a number below B·N. s4_x2's practised step,
    `B[STA.S T.lt.pair.AX SEL.S]`, is the chain `B[link]` the book's law teaches;
  - x + d: add on (x, d), checked by red and lt.

  Each needs the learner's addition, so the row and the step wait for it: level 3 instead of 0 and 1 in s4_x2's setting.
  That likely costs no level. B2rd's row is the book's, practised at level 3, and B2rd got the round at level 5. So did
  s4_x2: the row at 0, the step at 1, the addition at 2, the multiply at 4, and x², its rest and the round at 5.
- **The reader has no legal version that helps.** To read mod(mul(X, X), N) off the rows it must evaluate a product and
  a reduction. Before the learner has a multiply, no routine can. Once it has one, the cuts the reading would choose
  are already practised. "Every answer is below N" is a fact the comparison reads, so the reduction's cut alone could
  be chosen legally. Without the reader the source poses the cuts of every operation: a book of cuts, a generated book
  of SP_ABLATION.md §5.2's kind. That is a question about its grammar (LEGAL_PLAN.md §3, question 4), not about
  arithmetic.
- **Cost.** Likely no level for the answers (above).
- **Cheapest test.** Locally and free: answer the four cuts both ways on the slice harness's draws, with exact routines.
  They must agree; only when each becomes posable can differ. The run test is s4_x2 with the cuts answered by the
  book's constructions: one B200, under 40 minutes, about $4. B2rd predicts the round at level 5.

#### 3.14 The rest

- **Dreams** (problems.py:319): the learner's random programs, run on the target's inputs: its routines, legal.
- **Choosing and searching:** lp, ct2a2, lpca, the MDL score and the looks, the consolidation's allocator, the guided
  search, SPELL, REST_DEEP and the deeper probes. They decide from counts of right rows and from search costs, with no
  arithmetic on values. The looks search the target with candidate routines: legal when that target is the data's (in
  the harness they search the small target).
- **The bridge's slices** (`bar3/bridge/slices.py`): Python's arithmetic, as BRIDGE.md §6 says. Their in-run
  counterpart is the generic file's if block (LEGAL_PLAN.md §4, step 2).

### 4. The composition route, legally

#### 4.1 Each step, and where its answers come from

B2rd: COMPOSE2.md and `scale/compose2/records/c2_b2_rd.log`. s4_x2: `scale/slice/records/s4_x2.log` as it stood when
this was written (its round right on both of H1's ladders to T = 64, `slice/logs/ladder_s4_x2.txt`).

| step | level, B2rd / s4_x2 | answers in the harness | in a legal run |
|---|---|---|---|
| comparison, link, addition | 0–2 / 0–2 | facts, identities, a check | the same |
| the row | 3, book / 0, cut | repeated addition / Python | repeated addition, after the addition |
| Horner's step | none / 1, cut | Python | red(x + d·N) = x, after the addition |
| the multiply | 4 / 4 | repeated addition | the same |
| x² as a value | 5 / 5 | the learner's multiply | the same, held at full size |
| x² singled out (LAWNEAR, SUBGOAL) | 5 / 5 | numpy's law test, small target | the chain's test, answer rows (§4.2) |
| the rest from x² | 5 / 5 (cost 5 / 4) | the small target: Python | the T = 1 rows at full size |
| the round | 5 / 5 | H1's 400 answer rows | the same |

Which of bar 3's flags carried B2rd (COMPOSE2.md, one flag off at a time): REST_DEEP, SPELL, LAWNEAR and SUBGOAL were
each needed; CLEAN_VALUES, ROWCHECK and RESTRICT_ROWS were not; TIEBREAK never acted. So the legal rebuild needs the
rest of the way, the law view and the deeper probe, and can drop the restrictions.

#### 4.2 The law test by the learner's routines, simulated (local, $0)

numpy emulates the two routines exactly: the chain (ten times, subtract N where the value is at least N) and the
comparison (is it below N?). The answers y are the data's, read. A value g passes "y = g mod N" when the chain's result
equals y digit for digit on every row where that result is below N, with at least 4 such rows.

| value g | rows where the chain reduces: harness's 400 / all 1,000 | equal to y there | verdict |
|---|---|---|---|
| x² | 8 / 20 | all | passes |
| x² + N | 8 / 20 | all | passes |
| 2x, x + N, N − x, 10x | 400 / 1,000 | none | fails |
| x² + x, x² + 1 | 8 / 20 | none | fails |
| x·N, x³ | 0 / 1–2 | none | too few rows |

- x² + N passes too, as under numpy's test: it is also one law from the answer.
- The link alone, one conditional subtraction, is all B2rd's library has. With it the rows are those with x² < 2·N: 2
  of the harness's 400, 5 of the 1,000. That is below a 4-row floor on 400 rows. So B2rd's library would need the
  chain practised (the book's red on x + d·N, or the slice's step), or all 1,000 T = 1 rows. s4_x2's library has the
  step.
- numpy's test sees every row; this one sees 2% of them. On H1 that is enough: a wrong value would have to match the
  answer on 8 random full-size rows.

#### 4.3 What it might cost

From B2rd's log on one B200; the full-size figures are estimates, not measured.
- **Goes:** the small target's own search, every level: 19–35 s in B2rd.
- **Comes:**
  - the law test. The round's search probes 8 rows (spcore.py:1252), too few to hold the 2% of rows where the chain is
    exact. So its cheap levels are searched again on all 400 answer rows: at level 4 in B2rd, 48, 1,732 and 55,820
    states at costs 1–3. Then their distinct values go through the chain. Likely seconds a level;
  - the values and rests, posed at full size: 16 places instead of 2–6, perhaps 2–3 times the cost per state. s4_x2's
    rest needed only the regular 1M probe; B2rd's needed the 10M deep probe, 26 s at the practice widths;
  - acceptance on the other 600 T = 1 rows: negligible.
- **Reach is the risk, not time.** At full size, states merge less. Look-alikes are rarer, so SPELL's blind spot may
  shrink, but levels may grow faster. Whether the 4- or 5-token reduction stays in reach is what test 2 measures.

### 5. Tests, cheapest first

0. **Done here, local, $0:** H1's rows (§2) and the law test's simulation (§4.2).
1. **Local, free, minutes:** the slice cuts answered both ways on the slice harness's draws (§3.13). It can only fail on
   layout: which register, which digit order.
2. **Offline on one B200, about $1–2: the decisive one.** It first needs a full-size target mode in a copy of the
   harness, a `RowTarget` beside `oracle.SmallTarget`, perhaps 200 lines: the target's root built from the T = 1 answer
   rows at full size; values and rests posed there; the law view by the learner's routines on all 400 answer rows;
   acceptance on the other 600 T = 1 rows. Then, from s4_x2's recorded library at the start of level 5:
   - the law test over the round search's states of cost ≤ 3: does it pick x², `LDA.X STA.Y T.mul.x,o.XY`?
   - the rest from x² at full size, 8 probes, the 1M probe and then REST_DEEP's 10M with SPELL 16: does it find
     `LDA.N P[BRG.X T.red.Bx+d.AN]` and accept it on the other rows?
   - the round from there: 3 tokens over the two routines.

   Pass: both found. On B2rd's library the same test needs the chain posed, or all 1,000 rows (§4.2). Prior art for an
   offline search on a recorded library: SP_BREADTH.md §4.1, about $1.6.
3. **The tiny core, one B200, about $4–5:** s4_x2's settings with the full-size target mode, the cuts answered by the
   book's constructions, LAWNEAR and SUBGOAL by routines: the legal composition route end to end. Prediction, if test 2
   passes: the round at level 5 or 6. The controls exist: s4_x2's and B2rd's records.
4. **Then LEGAL_PLAN.md §4's order:** the route in an sp2-style file, its tiny core, one official H100 run (about $4).

### 6. A question for the judge

**Q5: may a decision about practice use Python's arithmetic on the training rows, if no target does?** It adds to
LEGAL_PLAN.md §3's four. Bar 3's decisions that would: the law view and LAWNEAR (§3.12), the slice reader (§3.13), the
restrictions' predicates (§3.3), the constants (§3.9). The reader is the sharpest case: it evaluates mod(mul(X, X), N)
on the labelled rows to choose what to practise, which is the task's own formula, run by hand-written code, one step
from a target. The project's own practice so far says no: fast2's miner computes other relations but feeds its lessons
only order and nonzero facts (REVIEW.md §5, rule 14), and BRIDGE.md §6 describes stage 3 as computing no number from a
value.
- **If yes:** these decisions move as they are, read on the T = 1 rows instead of the small target. Only §3's targets
  and inputs need their redesigns.
- **If no:** the law view and LAWNEAR take their routine versions (§3.12, §4.2), the restrictions and constants are
  dropped, and the slice source becomes a book of cuts.

### 7. The local checks

Short scripts over `train.jsonl`, run with `~/github/sotaku/venv/bin/python -I` from a scratch folder (this audit
edits only this file), in seconds and well under 1 GB:
- **the rows** (§2): the train rows grouped by (N, T); for each x, whether x + 1, x + 2, 2x + d (d < 2) and 10x + d
  (d < 10) are among its group's x, any of them and all of them; the same (N, x) across T; x² < k·N for k = 1, 2, 10,
  11, 100; classes by (x² mod N, N) at each T;
- **the harness's draw:** `core_c3.load_rows(400, seed=1, T=1)`'s rows, rebuilt with the same torch generator;
- **the law test** (§4.2), whose core is:

```python
v = g.copy()                              # g: a candidate's value on the rows; N: the moduli; y: the data's answers
for _ in range(10):                       # the chain: ten conditional subtractions
    v = np.where(v >= N, v - N, v)
exact = v < N                             # the comparison: the rows where the chain is exact (g < 11 N)
passes = exact.sum() >= 4 and (v[exact] == y[exact]).all()
```
