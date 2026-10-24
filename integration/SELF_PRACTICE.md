# Self-posed practice in stage 3: does the round still come without our curriculum?

*Written 2026-10-23. Code in `cpu_first/integration/self_practice/`, records in `integration/self_practice/records/`
(the evaluator's runs in `records/runner/`). Nothing submitted, no certified file touched (the
builder asserts their md5s).*

## In plain words

<!-- PLAIN -->
**What changed.** Stage 3 (s3i) learned x^2 mod N in one training run, but from nine practice problems a person chose
(the comparison, the conditional subtraction, the addition, the chain, the multiply's row, the multiply-accumulate, the
multiply, the long-division step, the reduction), in an order a person set, each also posed with two registers
swapped, and with the multiply's row posed so that it would be learned on its own. Here that list is gone. The learner
poses its own practice problems: every law of a small book (compare, equal, min, max, remainder, quotient, add,
subtract, multiply, multiply-add, add three) applied to numbers built from the data's own N and x in 8 ways, with the
inputs in every placement in the registers. That is about 70 problems to start and 36-180 at each later level, most of
them useless. A general rule picks which to practise; nothing says which laws matter, in which order, or where. The
certification is the same gate.

**Does stage 3 still discover the round when it picks its own practice? Yes.** On the untouched official manifest on an
H100 the file certifies: T = 64 on both ladders, every rung 768/768, every split exact, the round found by 746 s and
the model frozen by 785 s of the 3,600 s budget. The round is stage 3's own (copy x, multiply, reduce), and every
routine in it is the learner's: a multiply-accumulate row identical to stage 3's (`LDH.Y CLK P[MUL.X INC]`), school
multiplication over it, a conditional subtraction, ten of those as a chain, and long division over the chain. On 8 B200
seeds that file certified 7; on seed 7 it never found the multiply (its deep searches went to placements where the
multiply costs one move more). A fix to how deep searches pick their problems and their routines
(`submission_stage3_sp2.py`: a concept is searched deep in all its placements at once) certifies on the official
manifest too and on all 8 seeds, seed 7 included, at a price in time (official H100: the round by 1,029 s; the slowest
seed by 1,775 s).

**At what cost?** No more time than the designed curriculum. s3i's own control run on an H100 the same day found its
round by 1,047 s and froze by 1,092 s; the first self-posing file by 746 s and 785 s, the fixed one by 1,029 s and
1,077 s. That comes from how the problems are searched (hundreds of them in a few batched searches), not from posing,
which costs about a second a level. It keeps more routines (41-47 against 16), and the file is bigger (249-251 KB
against 174 KB).

**What does it pose?** Every one of stage 3's nine problems shows up among its own candidates, in every placement, and
the certified run practised all nine by level 4, in the order their constructions allow (nothing tells it that the
multiply's row comes before the multiply), plus some stage 3 never had (min, and subtraction on x + N).

**What failed, and how hard was it tried?** Bar 3's own choosing rules do not get there. Hardest-first and the
cheaper-pairs signal climbed 7-8 levels each (1.5-1.9 hours on a B200) without the round, and hardest-first in the
evaluator climbed 13 levels in 6,247 s without it. The cause is the registers: a problem's cost depends on where its
inputs sit, so "hardest first" practises the most awkwardly placed copies of easy laws, while the reduction needs cheap
pieces in particular registers. Rerun with every later fix, hardest-first never even got the addition at 8 probes, and
at 16 probes it got the multiply only at level 8 and no reduction by level 10 (3,976 s); the cheaper-pairs signal got
the multiply but no reduction. The rule that works treats a law on a pattern as one concept, rates it by its cheapest
placement, and practises it once in its cleanest placements. Two limits of reach had to be fixed too (a focused deep
search for the multiply; searching the round by routine calls once the library is big). Bar 3's target-derived sources
(the target's restrictions, projections, near misses and "the rest of the way") were tried in a study-only arm and only
took picks. In all: 43 tiny-core runs (about 20 B200-hours), 6 relaxed in-run attempts, and 27 evaluator records (two
files on the official manifest and 8 seeds each, the other formulas), about 40 GPU-hours.

**The other formulas** (INTEGRATION.md §1.3): none learned, as with s3i. The climb is the same on every formula (it
reads only N and x), and the round's search finds nothing: the formulas need a constant (y^2 + 1, + 3, - 1, 10 y^2 + 1),
another reduction (Montgomery's), more routine calls than the search allows (10 y^2 at 3 calls, y^3 at 4), or answers at
T = 8 that the search by calls does not take (M4). Picking its own practice does not lift that limit; the limit is in
the round's search and the law book.

**Still designed:** the law book (which 11 laws, which 8 ways to build numbers), the choosing rule (picked after bar
3's failed, by watching whether the round came), the machine, the search sizes, where the round is looked for, and the
rest of stage 3 (the reader, the cells, the countdown, the tournament).
<!-- /PLAIN -->

## 1. What was designed in stage 3, and what replaces it

Stage 3 (s3i, [INTEGRATION.md](INTEGRATION.md) §3) climbs a library of programs from nine practice problems a person
chose. The brief named six designed things; each is replaced by a generic rule, written once and applied to every law
alike:

| designed in stage 3 (INTEGRATION.md §4, the curriculum row) | here |
|---|---|
| which laws become practice problems: nine tasks (lt, modx, add, modB, mrow, mac, mul, step, reduce) | every law of a book of 11 operations on 8 patterns of the data's numbers, in every binding of its operands to the registers: 71-72 candidates at level 0, 100-230 a level once the addition exists |
| their order: the helper order (`NEEDS`), and since s3f the pieces first (mul after mac, reduce after modB) | none. A law is posed as soon as the routines its construction needs exist (an addition to build x + d N; the red and lt routines to check an addition), and the choosing rule picks among whatever is posed |
| the register variants: every task also posed with X and Y swapped | every injective binding of a law's operands to A, X, Y and N (12 for two operands, 24 for three) |
| mac posed with the running sum as its addend (A + x d), so that the multiply's row is a problem of its own | a + b c is one law of the book (mul.ladder's own form, the multiply cell's law), posed in its 24 bindings like every other; nothing marks the accumulator |
| widths, settings and probe counts | the operands' width by one rule (the widest whose instance fits the registers); the settings (10, 1) and (10, 2), the verify setting (10, 3) and the probe count (8 a setting, half of stage 3's 16) are kept as search sizes |
| the round's answers taken from the rows with T = 1 | the rows of the smallest T the batches have shown (general lessons v2's rule); a T above 1 is searched by running each candidate T times |

## 2. The learner on the digit machine (`spcore.py`)

Bar 3's learner ([PRACTICE.md](../bar3/PRACTICE.md), [BREADTH.md](../bar3/BREADTH.md)) ran on a bare counting machine,
where its law book could compute every target with numpy. Here it runs on libcore's digit machine inside the training
run, so three things had to change: where targets come from, how candidates are probed, and how a stage-3 file holds it.

### 2.1 The law book

Data terms, per probe, from a row's top parts at the pattern's width (libcore's `top_parts`): x' and N' (so x' <= N',
equal exactly when their digit strings are), another row's x' (o), the digits 0-9 over the probes (d), and zero.
Patterns build numbers from these by the division law's form a + q b (q one, a digit, or another row's x; b the modulus
or the base):

| pattern | instance kinds (one per operand) | laws posed on it |
|---|---|---|
| pair | (x', N'), (N', x'), (x', x'), (x', 0) | lt (borrow), eq, min, max, red (x' mod N'), add, sub |
| x+n | (x' + N', N') | red, sub |
| x+dn | (x' + d N', N') | red, div, lt |
| x+on | (x' + o N', N') | red, div |
| Bx+d | (10 x' + d, N'), x' != N' | red |
| x,d | (x', d) | mul |
| x,o | (x', o) | mul |
| o,x,d | (o, x', d) | a + b c, a + b + c |

Each is posed in every binding of its operands to A, X, Y and N; the other registers start with random digits (libcore's
dirty start), and the answer is A (a number) or the flag c (a comparison).

### 2.2 Targets without hand-written arithmetic

A training file may not compute a training target with Python's arithmetic (rule 14; stage 3 never does). So every
target is a fact or the learner's own routine:
- facts: x' <= N' (the mined x < N on top parts) and digit-string equality give lt, eq, min, max on the pair pattern,
  red's x' or 0, div's quotient d or o;
- identities: red(a + q N) = a for a < N (red.invariant) on every composite pattern;
- the learner's routines: a practised addition builds x' + N', x' + d N' (d additions) and x' + o N' (o additions), and
  the multiply's targets (d or o additions, mul.ladder and mul.distribute); a practised reduction on x' + d N' gives the
  step's target; the addition itself is checked by sub.inverse (red(a + N) = a and a + N >= N) with a practised red and
  lt.

So a law is posable only once the routines its construction calls exist: the order the stage-3 curriculum wrote down by
hand (`NEEDS`) is here a consequence of each law's construction, and nothing says which of the posable laws to practise.

A study-only oracle (`oracle.py`) poses the target itself at the practice widths, x'^2 mod N' with Python's integers,
so that bar 3's target-derived sources (restrictions, projections, near misses and "the rest of the way") can be tested
in the tiny core. It is never part of a training file.

### 2.3 Probes: many goals, many roots

Every law posed on the same pattern and binding shares one root (the same probes, the same dirty start), so one
breadth-first search answers all of them (`bfs_multi`: libcore's search with a goal per law, each goal's goal states
listed one spelling each and tested by its acceptor). And many roots share one search (`bfs_roots`): each keeps its own
budget and its own states (a per-root salt in the state key), and a level's work runs in one batched call. On the
tests the batched search gives the same found cost and the same accepted programs for every law as one search per root
(16 of 16, with and without budget streaming); on a B200 it took a 122-root level from about 10 minutes to about 70 s.

Each level, per root (in groups of 16):
1. the composition vocabulary (the routines and the 32 primitives, loops of bodies up to 3) at 1M stored states;
2. the first time a root is posed, its open laws over the primitives alone at 1M (that search never changes; it is
   where the cost-5 multiply-accumulate row comes from);
3. the 8 roots whose open laws came closest (below) get a deep probe at 5.8M (stage 3's measured 10-minute cap), over
   the primitives and only the routines the closest attempt at that root used (so a big library does not crowd it out:
   §5, item 7). In `submission_stage3_sp2.py`: every binding of the closest concepts, with the routines any of their
   closest attempts used and the routines those call (§5, item 10).

Closeness is the learner's own signal: how much of the gap left by the best single instruction the closest held state
closes (a law whose goal states acceptance rejected is not counted: it is reached, not out of reach).

### 2.4 Choosing, practising, and the round

The probes sort the pool as bar 3 does: trivial (cost <= 1), learnable (cost >= 2), alias (does on random machines
what a token does), duplicate (what an earlier learnable does), unsolved. Bar 3's rules:
- `lp` (hardest first): the K = 4 costliest learnable, ties in pool order;
- `ct2a2` (the cheaper pairs signal, bar3/breadth/cheap.py): the target searched with every learnable as a routine,
  and with every pair whose member is among the 2 hardest; the best pairs' members first.

and the rule this study adds (§3.4), `lpca`: rank concepts (a law on a pattern) by their cheapest binding, practise each
once in every binding tied for cleanest.

Practice is stage 3's sleep: the picks' programs join the corpus, BPE merges recurring pairs, and each pick becomes a
routine `T.<law>.<pattern>.<binding>`, its program chosen among the shortest by bar 3's rule (leave the inputs as they
were, change the fewest registers), then Occam (fewest primitives, then fewest steps executed). Then the round is
searched as stage 3 searches it (the routines and the straight-line primitives up to cost 5, 4M states a level) and,
once the library passes 12 routines and that finds nothing, by calls (§3.4).

## 3. The tiny core

The climb alone, offline, as `lib/profile_climb.py` runs stage 3's: exact cells, the practice problems from 4,096 rows'
top parts at widths 1-2 (settings (10, 1) and (10, 2), acceptance at (10, 3) on 60 fresh rows), the round at full
size against 400 rows with T = 1, up to 16 levels, one B200 each (`core.py` on Modal). Two arms:
- **the law-book arm**: the in-run setting exactly (the law book is the only source; the target is the round);
- **the oracle arm**: all of bar 3's sources (the law book, the target's restrictions and projections, its near misses'
  values and "the rest of the way"), with the target at the practice widths (x'^2 mod N', `oracle.py`).

### 3.1 What it poses

Level 0 poses 71-72 candidates on 12 roots (the pair pattern's 7 laws in 12 bindings, minus those that say nothing);
once an addition is practised, every pattern is posable: about 150-230 candidates on 90-120 roots a level (fewer under
`lpca`, which stops posing a practised concept's other bindings). Every one of stage 3's nine tasks has its law-book
counterpart in the pool (in every binding):

| stage 3's task | the learner's own candidate |
|---|---|
| lt | lt on the pair pattern (the borrow) |
| modx | red on the pair pattern |
| add | add on the pair pattern (checked by red and lt) |
| modB | red on x' + d N' |
| mrow | mul on (x', d) |
| mac | a + b c on (o, x', d) |
| mul | mul on (x', o) |
| step | red on 10 x' + d (posed once a red on x' + d N' is practised) |
| reduce | red on x' + o N' |

So the sources pose every piece. Whether a run gets the round depends on which bindings it practises.

### 3.2 The runs

Every run below is the law-book arm unless it says oracle; the times are a B200's. "Practised" lists which of stage 3's
nine pieces the run practised in some binding (`summarize.py TAG --pieces` gives the level, the binding and the program
of each).

<!-- CORE -->
| rule | run | levels | seconds | practised (of stage 3's nine) | the round |
|---|---|---|---|---|---|
| hardest first (`lp`) | core_lp_v7 | 8 | 6,693 | lt, modx, add, modB, mrow, mac, mul (level 4), reduce (level 7, in a binding nothing can call) | no |
| hardest first, 8 picks a level | core_lp8_v7 | 4 (stopped) | 1,168 | lt, modx, add, modB, mac | no |
| the cheaper pairs signal (`ct2a2`) | core_ct2a2_v7 | 7 | 5,534 | lt, modx, add | no |
| every learnable (`all`, bar 3's reference) | core_all_v7 | 3 (stopped) | 1,097 | lt, modx, add, each in all its bindings | no |
| unlock first (`lpu`) | core_lpu_v7 | 7 | 4,886 | lt, modx, add, modB, mrow, mac, mul | no |
| hardest first, oracle arm | small_lp_v7 | 7 | 6,125 | lt, modx, add, modB | no (the small target either) |
| cheaper pairs, oracle arm | small_ct2a2_v7 | 3 (stopped) | 3,482 | lt, modx | no |
| concepts (`lpca`), before the reach fixes | core_lpca_v7 | 6 | 3,435 | lt, modx, add, modB, mrow, mac, step, reduce | no: no multiply (§5, item 7) |
| concepts, all fixes, 16 probes a setting | core_lpca_v11 | 6 | 920 | all nine | **yes**, after level 5 |
| concepts, all fixes, 8 probes a setting | core_lpca_v11f | 5 | 657 | all nine | **yes**, after level 4 |
| hardest first, all fixes, 8 probes | core_lp_v12 | 10 | 427 | lt, modx | no: never the addition |
| hardest first, all fixes, 16 probes | core_lp_v13 | 10 (stopped) | 3,976 | lt, modx, add, modB, mrow, mac, mul (level 8) | no |
| cheaper pairs, all fixes, 8 probes | core_ct2a2_v12 | 7 (stopped) | 1,799 | lt, modx, add, mac, mul | no |
| concepts, oracle arm, all fixes | small_lpca_v12 | 6 (stopped) | 1,292 | lt, modx, add, modB, mrow, mac, mul | no (the small target either) |
| concepts, whole-concept deep probes (§5, item 10) | core_lpca_v13 | 5 | 693 | all nine | **yes**, after level 4 |
<!-- /CORE -->

The v7 runs were all stopped by hand, at the level shown ("stopped": in its first levels), to free their GPUs for the
next version; none had found the round. v11, v11f and lpca_v13 ended when the round was found, lp_v12 at its level
cap; the other v12-v13 runs were stopped by hand (§3.2 below).

**Bar 3's rules.**
- Hardest first and the cheaper pairs signal both learn the comparison (level 0), a conditional subtraction (level 1)
  and the addition (level 2); hardest first goes on to stage 3's multiply-accumulate row exactly (`LDH.Y CLK P[MUL.X
  INC]`, level 3) and the school multiply over its own row (`CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]`, level 4).
- Neither practises a reduction anything can call. Hardest first ranks the costliest learnable candidate first, and the
  cost of a candidate is inflated by awkward bindings (an operand in A or N needs moves): it practises variants of
  comparisons, `eq` and `a + b + c` at cost 5 before the cheap pieces the reduction needs. Those are register-specific:
  the reduction's loop keeps its dividend in X (`BRG.X` brings X's top digit in), so its chain must not use X, and must
  take the modulus from N or Y. The comparisons hardest first picks first (cost 5) read X, so every link and chain built
  on them uses X as scratch. Its one full reduction (level 7, `T.mul.x,d.YN W[LDA.X SHR.A T.red.x+dn.YA]`) takes its
  dividend in N, which no program can write.
- The cheaper pairs signal has nothing to go on at full size: the round is reached or not, and no lookahead with a
  pair of candidates reaches it until both the multiply and the reduction exist. It spends its levels on comparisons.

<!-- BAR3 -->
**Bar 3's rules with every fix of §3.4** (the focused deep probe and the round by calls), to see whether the fixes
rather than the concept rule made the difference:
- hardest first, 8 probes a setting (`core_lp_v12`): 10 levels in 427 s, and never the addition, so nothing past the
  pair pattern was ever posed. Its first reds (levels 1-4) are the costliest programs right on the pair pattern, and
  those compute "x' if x' < N', else 0" (`STA.S STA.Y T.lt.pair.YX CLA SEL.S`): on x' <= N' that is the remainder, but
  it cannot confirm an adder (red(a + N; N) = a on 4% of cases, `check_reds.py`). The two reds that do subtract
  (levels 5-6, `T.lt.pair.AN T.min.pair.AX`) take a minimum against X, a register they do not own: they passed their 8
  probes a setting but are right on only 69% of the narrowest setting's cases, and the addition's check needs all.
  So halving the probes hurts hardest first more than it hurts the concept rule, whose cleanest programs touch no
  register they do not own;
- the cheaper pairs signal, 8 probes a setting (`core_ct2a2_v12`): with the fixes it got further than before (stage 3's
  mac at level 3, the school multiply at level 4), but never the chain, the step or the reduction: levels 5-6 went to
  `eq` in four bindings; 7 levels in 1,799 s, stopped there to free its GPU;
- hardest first, 16 probes a setting (`core_lp_v13`): the addition at level 2 and stage 3's mac at level 3, then
  levels 4-7 on red over x' + N' and chains in X- and N-scratch bindings, the multiply only at level 8, and by level 10
  (3,976 s, past the whole in-run budget) no step and no reduction; stopped there.

**The oracle arm with the concept rule** (`small_lpca_v12`): the same pieces as the law-book arm (lt, the link and the
addition at levels 0-2, mac at 3, the multiply at 4), but two picks a level went to the target's values (`val.N/tgt`:
a value a near miss of the target holds partway through, bar 3's value source, posed as a problem of its own), and the
chain and the row came at level 5 instead of 3 and 4; by level 6 (1,292 s) neither the reduction nor the small target,
stopped there. The target-derived problems took picks and added nothing the round needs.
<!-- /BAR3 -->

### 3.3 The oracle arm

Bar 3's target-derived sources pose nothing stage 3 needs here. The target's near misses at the practice widths are
small-N shortcuts ("x div 10", right on 9 to 13 of 32 probes), whose values and "rest of the way" are more problems of
the same kind; its restrictions and projections are problems the comparisons already answer. On bar 3's counting
machine the square was a 3-instruction near miss; here x'^2 needs the multiply, and until the multiply is practised no
near miss holds it. With hardest first the oracle arm never practised the multiply-accumulate row or the multiply (7
levels, 6,125 s); with the cheaper pairs it was slower still (its lookahead runs every pair against the small target:
3 levels in 3,482 s).

### 3.4 Generic additions, and the round

- **Unlock first** (`lpu`): first the learnable candidates whose routine would let the learner pose a kind of problem
  it cannot pose now (its routine tried in every other construction), then hardest first. It practises the chain as
  soon as it is posable (level 3), but in a binding whose minimal programs use X as scratch, which the reduction
  cannot call; 7 levels, no reduction.
- **Concepts** (`lpca`): a concept is a law on a pattern; its hardness is the cost of its cheapest binding, so an
  awkward binding does not make it look hard; each concept is practised once, in every binding tied for cleanest
  (fewest registers touched besides A: bar 3's rule for routines, applied to the choice of binding; the generated
  counterpart of stage 3's X/Y variants), and a practised concept's other bindings are not posed again unless no new
  concept is learnable. It practised the comparison, the conditional subtraction and the addition in A against X, Y
  and N (levels 0-2), the chain in the same three and the multiply-accumulate row in its four A-accumulator bindings
  (level 3), then the full reduction, long division over its own chain (level 4).
- **Two more obstacles, both of the big library** (lpca keeps about 3 routines a concept: 44 routines and 27 merges by
  level 4):
  - the multiply (`CLA W[mac INC SHL.X]`, cost 5) was out of even the deep probe's reach: with 80 tokens its level 4
    passes 5.8M states. Fix: a deep probe searches with the primitives and only the routines its root's closest attempt
    used (here `CLA T.acc.mul.o,x,d.AXY`, right where o has one digit): 33 tokens, the multiply found in 7 s;
  - the round (cost 6 with this reduction, whose modulus is in A) was out of the round search's reach (stage 3's: the
    routines and the straight-line primitives, 4M states a level, reaching cost 4 with 75 tokens). Fix: past 12
    routines, when that search finds nothing, the round is also searched by calls: one item per routine and per routing
    of its inputs (each from A, X, Y, N or zero, by straight-line moves) plus the instructions that act on whole
    registers, up to 2 calls (tested on s3f's routines first: it finds s3f's round as 2 calls).

**The round, discovered.** `core_lpca_v11f` (lpca, 8 probes a setting, all the fixes above), level 4, 657 s on a B200:

    LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA        x into Y, multiply, the product into X, N into A, reduce

with every routine it calls found and practised by the learner from its own law book:

| routine | program | what it is |
|---|---|---|
| T.lt.pair.AY | `CLC P[SBB.Y INC]` | A - Y digit by digit: the ripple (level 0) |
| T.red.pair.AY | `STA.S T.lt.pair.AY SEL.S` | subtract Y unless that borrows: the link (level 1) |
| T.red.x+dn.AY | `B[T.red.pair.AY]` | ten links: the chain, A mod Y for A < 10 Y (level 3) |
| T.acc.mul.o,x,d.AXY | `LDH.Y CLK P[MUL.X INC]` | A + X times Y's digit: stage 3's multiply-accumulate row exactly (level 3) |
| T.mul.x,o.XY | `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]` | X times Y: school multiplication over its own row (level 4) |
| T.red.x+on.XA | `STA.Y P[BRG.X T.red.x+dn.AY]` | X mod A: the modulus into Y, then bring X's digits in one at a time, the chain after each: long division (level 4) |

The round reproduces all 400 answer rows at full size (16 places). It is stage 3's round up to the registers: stage 3
reduced on Y with the modulus in N; this learner's reduction takes the modulus in A, which the round loads from N. The
same run with 16 probes a setting (`core_lpca_v11`) had the multiply at the same level (in the other X/Y binding) and the
reduction a level later: the round after level 5, 920 s.

## 4. Stage 3 with self-posed practice (`submission_stage3_sp.py`, `submission_stage3_sp2.py`)

### 4.1 The file

s3i's template with its climb (`lib.libcore.Climb` over the nine designed tasks) swapped for `spcore.SelfClimb`,
inlined after libcore (`make_sp.py` builds it: 248,570 bytes, s3i 173,817). Everything else is s3i's: the reader, the
learned cells, the round's acceptance, the countdown over T, the tournament. The climb does one unit of work a training
step (pose, a group's probe, a group's deep probe, classify, choose, sleep, the round), each under the training
deadline. The answers the round is searched against are the rows of the smallest T the batches have shown, kept from
phase C on. Two builds (§7): `submission_stage3_sp.py` (f0ef5b06), the first certified, and `submission_stage3_sp2.py`
(afc70fe9), whose deep probes take whole concepts (§5, item 10). Settings (`SP_*` at the top of the file): `lpca` with
4 concepts a level (each in its tied-cleanest bindings), 8 probes a setting, probes of 1M, deep probes of 5.8M for 8
roots a level, the round's flat search of 4M states to cost 5, its search by calls past 12 routines to 2 calls.

### 4.2 Results on H1

<!-- RUNS -->
Every certified run below is exact everywhere: T = 64 on both ladders, every rung 768/768, and every split (ood_n_t
10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999). The times are seconds into training (the climb starts after the
last progress line before its first unit; the round and the freeze by the first progress line after them, every 100
steps); "training" runs to step 20,000, most of it after the freeze. `summarize.py --runs` prints the full table.

**The file as first certified** (`submission_stage3_sp.py`, md5 f0ef5b06):

| run | device | seed | max T, seen / unseen moduli | climb from / round by / frozen by | training (frozen at step) | evaluation |
|---|---|---|---|---|---|---|
| **official manifest, untouched** | H100 | 74 | **64 / 64** | 66 / 746 / 785 | 938 (883) | 154 |
| official manifest | B200 | 74 | 64 / 64 | 45 / 588 / 615 | 730 (883) | 91 |
| official manifest | B200 | 1 | 64 / 64 | 40 / 572 / 589 | 709 (825) | 97 |
| official manifest | B200 | 2 | 64 / 64 | 46 / 688 / 714 | 833 (886) | 98 |
| official manifest | B200 | 3 | 64 / 64 | 76 / 749 / 795 | 987 (795) | 199 |
| official manifest | B200 | 4 | 64 / 64 | 50 / 626 / 654 | 774 (891) | 94 |
| official manifest | B200 | 5 | 64 / 64 | 59 / 799 / 827 | 989 (842) | 154 |
| official manifest | B200 | 6 | 64 / 64 | 43 / 520 / 582 | 700 (793) | 98 |
| official manifest | B200 | 7 | **not certified** (test 7 / 9,999) | 49 / - / - | 3,600 (never frozen) | 7 |
| relaxed budget (36,000 s), sp6 = the same climb, md5 e46c48a1 | B200 | 74 | 64 / 64 | 88 / 810 / 852 | 1,080 (883) | 179 |

So it certifies on the untouched official manifest and on 7 of 8 seeds. Seed 7 never found the multiply (§5, item 10).

**The fixed file** (`submission_stage3_sp2.py`, md5 afc70fe9: the deep probes take whole concepts, §5 item 10):

| run | device | seed | max T, seen / unseen moduli | climb from / round by / frozen by | training (frozen at step) | evaluation |
|---|---|---|---|---|---|---|
| **official manifest, untouched** | H100 | 74 | **64 / 64** | 84 / 1,029 / 1,077 | 1,277 (887) | 226 |
| official manifest | B200 | 74 | 64 / 64 | 45 / 702 / 729 | 855 (887) | 97 |
| official manifest | B200 | 1 | 64 / 64 | 39 / 650 / 666 | 785 (829) | 94 |
| official manifest | B200 | 2 | 64 / 64 | 50 / 744 / 770 | 886 (882) | 97 |
| official manifest | B200 | 3 | 64 / 64 | 65 / 1,775 / 1,804 | 2,010 (809) | 209 |
| official manifest | B200 | 4 | 64 / 64 | 90 / 937 / 981 | 1,209 (895) | 197 |
| official manifest | B200 | 5 | 64 / 64 | 41 / 651 / 669 | 784 (836) | 92 |
| official manifest | B200 | 6 | 64 / 64 | 64 / 818 / 897 | 1,073 (799) | 155 |
| official manifest | B200 | 7 | 64 / 64 | 72 / 931 / 969 | 1,187 (883) | 161 |

It certifies on the untouched official manifest and on every seed, seed 7 included, at a price in time: deep probes
over whole concepts cost more (on seed 3, 12 bindings of the reduction's pattern with a large shared vocabulary took
817 s at level 5), so its round comes later than the first file's (official H100: 1,029 s against 746 s; the B200
seeds: 650-1,775 s against 520-799 s). Every run of both files chose the same round, `LDA.X STA.Y T.mul.x,o.XY STA.X
LDA.N T.red.x+on.XA` or its X/Y twin (or both, the tournament splitting the weight).
<!-- /RUNS -->

### 4.3 What the certified run posed and practised

The official run's climb (f0ef5b06), level by level (steps 641-684). The other seeds' climbs take the same course (the
same pieces at the same levels, a few picks more or fewer), except seed 2 (the round a level later) and seed 7 (§5,
item 10):

<!-- LEVELS -->
| level | posed (candidates on roots) | learnable / duplicate / trivial / unsolved | practised (cost, program) | seconds |
|---|---|---|---|---|
| 0 | 71 on 12 | 14 / 4 / 0 / 53 | lt in A against X, Y, N (4, `CLC P[SBB.X INC]`); sub in A against Y (4, `CLC P[INC SBB.Y]`) | 28 |
| 1 | 48 on 12 | 21 / 3 / 0 / 24 | min in A against X, Y, N (4, `STA.S T.lt.pair.AX LDA.X SEL.S`); red in A against X, Y, N (3, `STA.S T.lt.pair.AX SEL.S`): the link | 56 |
| 2 | 36 on 12 | 5 / 3 / 0 / 28 | add in A against X, Y, N (4, `CLC P[ADC.X INC]`) | 28 |
| 3 | 177 on 96 | 39 / 26 / 6 / 106 | a + b c in its four A-accumulator bindings (5, `LDH.Y CLK P[MUL.X INC]`): stage 3's mac; red on x' + N' in A against X, N (2, `P[T.red.pair.AX]`); sub on x' + N' in six bindings (2, `LDA.X T.lt.pair.AY`); red on x' + d N' in A against X, Y, N (2, `B[T.red.pair.AX]`): the chain | 232 |
| 4 | 120 on 96 | 27 / 20 / 3 / 70 | mul on (x', o) in X, Y both ways (5, `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]`): the multiply; red on x' + o N' in X against A (4, `STA.Y P[BRG.X T.red.x+dn.AY]`): the reduction; red on 10 x' + d in six bindings (2, `LDA.X T.red.x+dn.AY`): the step; mul on (x', d) in four bindings (2, `CLA T.acc.mul.o,x,d.AXY`): the row | 318 |
| round | by calls, 2 calls | | `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA` and its X/Y twin, right on every answer row | 9 (level 4's search; every level's together: 46, in the seconds above) |
<!-- /LEVELS -->

All nine of stage 3's pieces were practised, by level 4, in the order their constructions allowed; 13 concepts in all.
Posed but not practised before the round: eq, max, the quotient (div on x' + d N' and on x' + o N'), lt on x' + d N',
and a + b + c. The learner also practised what stage 3 never posed (min, sub, red and sub on x' + N'): 41 routines and
13 merges at the round, where s3i has 16 routines (its nine tasks and their X/Y twins) and 24 merges. The climb's 662 s
went mostly to probes (549 s); deep probes took 63 s, the round's searches 46 s, posing and choosing 4 s.

### 4.4 Cost against s3i

<!-- COST -->
The same day, on H100s, seed 74, the official manifest: s3i's own control run (the ablation study's
`integration/ablation_s3/records/control_h100_seed74.json`, file md5 0e813144, read here, not rerun) against this
study's two files' official runs.

| | s3i: the designed curriculum | self-posed (f0ef5b06) | self-posed, whole-concept deep probes (afc70fe9) |
|---|---|---|---|
| the climb starts after | 78 s | 66 s | 84 s |
| the round found by | 1,047 s (step 671) | 746 s (step 684) | 1,029 s (step 684) |
| frozen by | 1,092 s | 785 s | 1,077 s |
| training, all of it | 1,286 s | 938 s | 1,277 s |
| evaluation | 195 s | 154 s | 226 s |
| levels to the round | 6 | 5 | 5 |
| practice problems searched | 16 (the nine and their twins), one search each, 5.8M states (the multiply's row: 27M) | 452 posed (71 + 48 + 36 + 177 + 120), 41 practised; probes of 1M for 16 roots at a time, 8 deep probes of 5.8M a level | as f0ef5b06, with the deep probes over every binding of the closest concepts |
| routines (merges) at the round | 16 (24) | 41 (13) | 47 (34) |
| seeds certified on the official manifest | 8 of 8 (H100s, INTEGRATION.md §1) | 7 of 8 (B200s) | 8 of 8 (B200s) |
| the file | 173,817 bytes | 248,570 bytes | 250,643 bytes |

Posing its own practice costs nothing in time here: the first file reaches the round about 300 s sooner than s3i, and
the fixed one about as soon (1,029 s against 1,047 s). That is not because posing is cheap (it costs about a second a
level) but because of how the problems are searched: hundreds of problems in a few batched searches of 1M states each,
and the 5.8M budget only for the 8 roots that came closest, where s3i searches each of its 16 problems alone at 5.8M
(some of its searches take 100-180 s). s3i would gain from the same batching; the comparison says the generic
curriculum is affordable, not that it is faster in itself.
<!-- /COST -->

### 4.5 The other formulas (INTEGRATION.md §1.3)

<!-- ACCEPT -->
The certified file (f0ef5b06) unchanged on each formula, as §1.3 runs them: H1's prompts with the formula's answers,
seed 74, an H100, at most 1,500 steps; M4 on its own manifest with §1.3's relaxed budget (3,600 s, 1,500 steps)
(`plans/sp_accept.json`). Next to it, the round's search offline (`accept_offline.py`): the routines the H1 run
discovered (its library at the round), each formula's answers on H1's T = 1 rows, the search as the file runs it (to
cost 5, then by calls to 2) and with one call more.

<!-- ACCEPT_TABLE -->
| formula | the evaluator's run (f0ef5b06, H100) | its climb | the round's search offline: as the file / with 3 calls |
|---|---|---|---|
| H1: x^(2^T) mod N | certified (64 / 64, §4.2) | 5 levels, the round at level 4 | 2 calls: `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA` / - |
| y^2 + 1 | not certified (no round in 3,600 s; test 8 / 9,999) | 10 levels, then nothing learnable | none / none |
| 10 y^2 | not certified (test 6 / 9,999) | 10 levels | none / **3 calls**: `LDA.X STA.Y T.mul.x,o.XY SHL.A STA.X LDA.N T.red.x+on.XA` |
| x^(3^T) (y^3) | not certified (test 11 / 9,999) | 10 levels | none / none |
| Montgomery: y^2 10^-8 mod N | not certified (test 10 / 9,999) | 10 levels | none / none |
| 10 y^2 + 1 (held out) | not certified (test 12 / 9,999) | 10 levels | none / none |
| y^2 + 3 (held out) | not certified (test 6 / 9,999) | 10 levels | none / none |
| y^2 - 1 (held out) | not certified (test 7 / 9,999) | 10 levels | none / none |
| M4 (practice tier, every T = 8) | not certified (relaxed 3,600 s; test 8 / 9,000) | 10 levels; its round searched with T = 8 (flat only) | - |

Every run's climb went on past the H1 round's pieces: a + b + c at level 5, nothing new at level 6, other bindings of
what it had at level 7, then nothing learnable among 48-141 candidates (levels 8-9): the law book was used up within
the probes' reach, and the round was searched again every 10 steps until the budget's margin.
<!-- /ACCEPT_TABLE -->

The climb is the same on every formula: it reads only the rows' N and x, so up to the first round search every run here
but M4 (its own rows) is the H1 run step for step (the same candidates, the same picks, the same multiply and reduction at level 4). Only the
round's search sees the answers, and with this learner's routines the only formula one step away is 10 y^2 (3 calls:
`LDA.X STA.Y T.mul.x,o.XY SHL.A STA.X LDA.N T.red.x+on.XA`), as for s3i (cost 6). The others need what no routine
holds: a constant (y^2 + 1, + 3, - 1, 10 y^2 + 1), a different reduction (Montgomery's 10^-8), or a reduction between
two multiplies (y^3: with 16 places a register holds x^2 but not x^3, so the cube needs multiply, reduce, multiply,
reduce, 4 calls; the 3-call search finds nothing). M4's rows all have T = 8, so its answers are searched by running
each candidate 8 times, which only the flat search does (cost 5), not the search by calls. So the self-posed curriculum
does not lift stage 3's limit on the other formulas: the limit is in the round's search (2 calls, no constants) and in
the law book (no law makes a constant), the same place as s3i's.

The fixed file (afc70fe9) was not run on the other formulas: it changes only which roots the deep probes take, the
round's search and the law book are the same, and the offline search above already uses the routines a certified run
discovered (the H1 runs of both files end with the same multiply and reduction).
<!-- /ACCEPT -->

## 5. What failed on the way, and how hard it was tried

Each of these was found in a run, reproduced or explained, fixed by a rule that holds for every law alike, and the
runs restarted (v1 to v13 of the tiny core, sp1 to sp6 in the evaluator; `records/` keeps each version's records):

1. **Reach (v1).** A probe of 1M stored states reaches cost 5 over the primitives alone (level 4 holds about 0.5M
   states, level 5 is streamed), but once a dozen routines are in the vocabulary its level 4 passes 2M and is streamed:
   only cost 4 is reached. The multiply-accumulate row (`LDH.Y CLK P[MUL.X INC]`, cost 5 over the primitives) and the
   multiply over it (cost 5 over the routines) were then out of every probe's reach, so never learnable, so never
   picked (v1, 5 levels on 2 rules). Probing every root at stage 3's 5.8M would cost about 1.5 hours a level here. Fix:
   a root's first probe also runs over the primitives alone (1M: it never changes, so once), and the 8 roots whose
   open laws came closest get a 5.8M probe.
2. **The gate (v3).** "Closest" as the share of probes the best state gets right put every deep probe on `eq`, whose
   answer is "unequal" on 90% of the probes, so a constant guesses most of it. Fix: closeness is the share of the gap
   left by the best single instruction that the probe closes, and a law whose goal states were all rejected by
   acceptance (reached, not out of reach) does not count. `eq` still often wins the gate: it is close, just useless.
3. **The alias (v4).** The pair pattern only shows x' <= N', so for red(x'; N') "x' if x' < N', else 0" passes as
   well as the conditional subtraction. Hardest first's first practised red was that alias in one run; the addition's
   check (red(a + N; N) = a) used the first practised red, so no true adder could pass, the patterns that need an
   addition were never posed, and the climb spent levels 2-5 on comparisons (v4, hardest first). Stage 3 met the same
   alias (s3c) and got past it by an Occam tie-break. Fix: a check passes when any practised routine of its law
   confirms it. Two follow-ons, found the same day: a check must use the library of the moment (a routine practised
   after a check was built crashed it, v6), and a cached rejection must be forgotten when a routine joins (else an
   adder rejected while only the alias existed stays rejected).
4. **The time limit (stage-3 file, local).** The shared vocabulary of a level (the routines with every loop over
   them) was built under one search's 900 s limit; on a slow host the first group of a level could spend the whole
   limit there and come back with nothing, which ended the climb ("nothing learnable"). Fix: vocabularies are built
   under the training's deadline, and each search's limit starts after.
5. **Bar 3's choosing rules (v1-v7, sp1-sp3).** Hardest first and the cheaper pairs signal never got a usable reduction
   (§3.2): 7-8 levels each in the tiny core (5,534-6,693 s), and in the evaluator hardest first climbed 13 levels in
   6,247 s on a relaxed budget (sp3) without the round (that file had neither the focused deep probe nor the search by
   calls; bar 3's rules with every fix: §3.2). Hardest first with 8 picks a level and bar 3's reference (`all`,
   every learnable) only grew the pool of variants faster (`all` practised 14-27 a level and was stopped in level 3).
   Diagnosis: the binding, not the law, decides a candidate's cost, so "hardest" means "most awkwardly placed". Fix:
   the concept rule (`lpca`, §3.4), which ranks a law on a pattern by its cheapest binding.
6. **Unlock first (v7).** A rule that prefers whatever lets the learner pose something new got the chain early, in a
   useless binding (§3.4). Kept as a rule (`lpu`), not used.
7. **The multiply out of reach (lpca v7-v10, sp5).** With lpca's bigger library (about 80 tokens by level 4), the
   multiply (cost 5 over its own row) passed even the deep probe's 5.8M states at level 4. In the evaluator, sp5 (lpca
   with the round's search by calls) practised the full reduction at level 4 and then found nothing learnable among 36
   candidates at level 5. Fix: the deep probe's vocabulary is the primitives plus the routines its root's closest
   attempt used (33 tokens; the multiply in 7 s).
8. **The round out of reach (lpca v7-v9).** Stage 3's round search (the routines and the straight-line primitives,
   4M states a level) reaches cost 4 with 75 tokens; this learner's round is cost 6 (its reduction takes the modulus in
   A, one more move). Fix: the round searched by calls (§3.4), tested first on s3f's routines.
9. **Bar 3's target-derived sources (oracle arm, v1-v7).** No help here (§3.3); not in the training file.

<!-- FAILED -->
10. **The multiply's cheap bindings never probed deep (seed 7 of 8, f0ef5b06).** On one seed the climb practised the
    multiply's row (mul on (x', d): `CLA T.acc.mul.o,x,d.AXY`) at level 4, before the multiply. From then on the
    deep probes' gate (the 8 roots closest to an open law) sent every deep probe to the multiply's A-bindings (x' or o
    in A: AY 0.71, NA 0.54, AX 0.50, YA 0.50, XA 0.46, AN 0.43), where the multiply needs one move more than its cost-5
    programs in X and Y; and each deep probe's vocabulary (the routines its root's closest attempt used) held the row,
    which clears A first, not the accumulating row the multiply loops over. At level 6 nothing new was learnable; levels
    7 on practised other bindings of what it had. Fix (`submission_stage3_sp2.py`, `SP_DEEP_BY = "bindings"`,
    `SP_DEEP_FOCUS = "closure"`): the deep probes take the closest concepts whole (every binding where the concept is
    still open, concept by concept until 8 roots are taken), with the routines any of the concept's closest attempts
    used and the routines those call. In the tiny core (`core_lpca_v13`) it reaches the round at level 4 in 693 s
    (v11f: 657 s), now with both X/Y multiplies; in the evaluator it certifies on the official manifest and on all 8
    seeds (§4.2), more slowly: a whole concept's deep probes with a shared vocabulary can take 800 s a level (seed 3).
11. **Bar 3's rules with every fix (v12-v13).** §3.2: hardest first never got the addition at 8 probes a setting (its
    subtractions read a register they do not own), and at 16 probes got the multiply at level 8 and no reduction by
    level 10 (3,976 s); the cheaper pairs signal got the multiply (level 4) but never the chain or the reduction in 7
    levels (1,799 s, levels 5-6 on `eq`). Both were stopped by hand to free their GPUs.
12. **The other formulas (§4.5).** Not a failure of the curriculum: the round's search and the law book, as in s3i.
<!-- /FAILED -->

## 6. What is still designed

- **The law book**: 11 laws and 8 patterns (§2.1). Which laws and which ways of building numbers from the data are in
  the book is a person's choice, though a generic one: each law is stated once, and no law or pattern names the round.
  The book is bigger than the round needs: the round calls none of eq, min, max, div, sub, add or a + b + c (the
  addition is needed to pose the problems the reduction comes from).
- **The choosing rule** (`lpca`), picked after bar 3's rules failed here, by watching whether the round came. That is
  one more round of design with the target in view; it holds for every law alike and names none.
- **The machine**: libcore's digit machine, its 32 primitives and loop forms, and the registers (as in stage 3).
- **The search sizes and rules**: probe and deep-probe budgets, 8 deep probes a level (sp2: whole concepts), 8 probes
  a setting, the round's 4M states, the search by calls past 12 routines to 2 calls (measured, or halved for speed;
  none is per law).
- **Where the round is looked for**: short combinations of the routines on the answers of the smallest T (as in stage
  3, with the smallest-T rule in place of T = 1).
- **The rest of stage 3**: the reader's grammar, the cells' laws, the countdown over T, the tournament.

## 7. Files and how to rerun

Everything is in `cpu_first/integration/self_practice/`; nothing outside it changed, and no certified file was
touched (`make_sp.py` asserts their md5s before it builds).
- `spcore.py`: the learner, torch only, so that a stage-3 file can inline it after libcore. The law book (`PATTERNS`,
  `LAW_BOOK`, `law_target`), the data's terms (`Draw`), roots and goals (`Root`, `Goal`, `Acceptor`), the searches
  (`bfs_roots`: many roots in one search; `bfs_multi`: one root, many goals), and the climb (`SelfClimb`, one unit of
  work per call: pose, probe, deep probes, classify, the rule, sleep, the round). Rules: `lp`, `ct2a2` (bar 3's),
  `all` (bar 3's reference), and the variants of §3 (`lpu`, `lpc`, `lpca`).
- `oracle.py`: the tiny core's oracle arm (the target at the practice widths, with Python's integers; never in a
  training file): the target's restrictions, projections, near misses and "the rest of the way".
- `core.py`: the tiny core, offline: the climb alone on the data's rows with exact cells (env: `RULE`, `K`, budgets,
  `TARGET`, `SOURCES`, ...); writes `records/<TAG>.json` as it goes.
- `modal_sp.py`: the tiny core's runs on Modal (B200; the container's 12 h timeout, a 64 GiB hard memory limit, the
  module's own deadline, and its process group killed with the call). Logs and records on the Volume `self-practice`.
- `stage3_sp_template.py`, `make_sp.py`: stage 3 (s3i's template) with the climb swapped for `SelfClimb`, the answers
  at the smallest T, and the climb's settings (`SP_*`). Two builds:
  - `submission_stage3_sp.py` (md5 f0ef5b06): the file certified on the official manifest (§4.2), built before the
    template had `SP_DEEP_BY` and `SP_DEEP_FOCUS` (it runs as "root" and True); its source is also in every record of
    its runs;
  - `submission_stage3_sp2.py` (md5 afc70fe9, `make_sp.py submission_stage3_sp2.py`): the same with the deep probes
    taking whole concepts and the routines' closure (§5, item 10).

  The evaluator's runs go through the shared runner (`tools/modal_runner.py`), plans in `plans/`, records in
  `records/runner/`.
- `smoke_phase_c.py`: a file's phase C locally (the reader and cells at the truth, tiny budgets; `FILE=` picks the
  build): the climb runs inside training, one unit per step.
- `check_reds.py`: which of a run's practised reds can confirm an addition (§3.2), locally in seconds.
- `accept_offline.py`: §1.3's formulas against the round's search with the certified run's routines, on H1's T = 1
  rows (§4.5).
- `test_round_t.py`: the round's search on answers of one T, with s3f's routines: T = 1 finds s3f's round at cost 5;
  T = 2 finds the round when it is one routine (run twice), and accepts nothing wrong; the search by calls finds s3f's
  round as 2 calls.
- `peek.py`, `summarize.py`: check-ins, and the tables here (`--picks`, `--pieces`, `--runs`).

`make_sp.py` builds `submission_stage3_sp2.py` (and refuses to overwrite the certified `submission_stage3_sp.py`); the
runner's `--seeds 8` is a count (74 and 1-7), `--seeds 7,` one seed.

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python integration/self_practice/make_sp.py
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file integration/self_practice/submission_stage3_sp2.py --out integration/self_practice/records/runner
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/self_practice/submission_stage3_sp2.py --seeds 8 --memory-gb 48 --out integration/self_practice/records/runner
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode plan --plan integration/self_practice/plans/sp_accept.json --out integration/self_practice/records/runner
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach integration/self_practice/modal_sp.py --envs '[{"TAG": "core_lpca", "RULE": "lpca", "NP": "8", "LEVELS": "16", "DEEP_BY": "bindings", "DEEP_FOCUS": "closure"}]'
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach integration/self_practice/modal_sp.py --envs '[{"TAG": "accept_offline", "MODULE": "integration.self_practice.accept_offline"}]'
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m integration.self_practice.check_reds core_lp_v12 core_lpca_v11f
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python integration/self_practice/summarize.py core_lpca_v11f --picks --pieces
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python integration/self_practice/summarize.py --runs
```
