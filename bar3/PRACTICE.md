# Choosing its own practice: can the learner invent x² mod N without our task list?

*Written 2026-10-23. Code in `cpu_first/bar3/`, records in `bar3/records/`. Nothing committed, nothing submitted, no
file outside `bar3/` changed.*

## In plain words

<!-- PLAIN -->
**The question.** In the inventing test ([INVENTING.md](../library/INVENTING.md)), a learner that searches for short
programs and turns what it solves into new instructions invented x² mod N (square x, keep the remainder after
dividing by N) on a bare counting machine, but only when we handed it our list of 17 practice problems: copy, add,
compare, "subtract N unless that goes below zero", multiply, square, and so on. Without the list it never did. Here
it has to choose its own practice.

**Where its practice problems come from.** Three sources that know nothing about the answer:
1. a general law book of arithmetic (add, subtract, multiply, divide, remainder, compare, gcd, ...) applied to the
   numbers in the data, for example "X mod N, where X is either x or x + N": 68 problems, most of them irrelevant;
2. changes of the target itself: only some of its inputs ("only where x² < N"), only part of its answer ("its last
   binary digit"), and its own best wrong attempts: for each, "what does this attempt compute?" and "what is left to
   do after it?";
3. random programs of its own (the inventing test's "dreams").

On the pairs with N ≤ 7 that makes 192 candidate problems. The learner tries each with a small search: 11 are
trivial, 46 it can solve and has not practiced yet ("learnable"), 33 repeat those, 102 are out of reach for now. Then
it picks four to practice, by one of several rules, and makes their solutions into single instructions.

**The answer: yes, at this tiny size.** It invents x² mod N in its first round, by the same algorithm it found with
our list: square x, then x times subtract N unless that goes below zero (4 instructions once both pieces are
instructions of their own; right on every pair up to N = 80). Three things were needed:
1. **A source that poses the missing piece**, the step "subtract N unless that goes below zero" (the *link*). The law
   book poses it, and so do the learner's own wrong attempts (the example below). Random programs and plain
   restrictions of the target never pose it, and without it nothing works.
2. **A choice that keeps it.** The most reliable signal is "which two problems, made into single instructions, make
   x² mod N shortest?". Of 1,035 pairs, only the pairs (square, link) make it short: 66 bits saved, against 16 for the
   best other pair. It held when the law book was doubled: of 5,050 pairs, the 4 that pair a square-like problem with
   a link came out far ahead, though choosing then took 29 search-hours instead of 6. Two simpler rules also worked:
   "practice the hardest problems you can solve" (the link is the hardest; with the doubled law book it took a second
   round) and "practice everything you can solve" (46 new instructions, or 101 with the doubled book). Choosing one
   problem at a time by the same signal failed (the square alone saves nothing), and so did choosing the most novel
   problems, and random picks. Wrong picks cost more than a wasted round: four useless new instructions made the link
   too expensive for the next round's quick searches, so those learners could not pick it up later.
3. **Routines without side effects.** Each link problem has six equally short programs, and the first one found uses
   N as scratch: it works once but cannot be repeated. Keeping the one that leaves its inputs as they were (a general
   rule for subroutines) was necessary: without it the three rules above all failed. Our curated run had been lucky
   here.

**A tiny example, with no law book at all.** The learner's quick search for x² mod N on the pairs with N ≤ 7 finds
that plain x² is right on 16 of the 28 pairs (its second-best attempt). From that attempt it poses "compute x²" and
"finish the job from x²". Finishing is too hard as posed, but on the easy inputs, where x² < 2N, it takes one step:
subtract N unless that goes below zero, a 6-instruction program found in 30 seconds of search. With both made single
instructions, x² mod N is 4: `T.val1 L.X[MOV.X.A T.res2]`, "x², then x times: subtract N unless negative". Of the 190
pairs of problems it could solve from its own attempts, this is the only one that makes x² mod N short.

**What it chose, compared with our list.** With all sources and the pairs signal it picked the square (from the law
book, "X·X"), the link twice (once from the law book, once from its own attempt at x²) and one irrelevant problem
("N div X"). Our list has 16 problems; the learner kept the 2 that the answer uses and did without the other 14 (add,
compare, multiply, the full reduction, ...). Its 46 learnable problems between them solved 12 of our 16 (and trivial
ones 2 more), so the sources were rich enough; the choice is what threw out the rest.

**The wrap-around.** It never invented "add one, wrap around at N", the step that made the inventing test's search
succeed in a second. Its substitute is the link, "wrap once", repeated x times after squaring. When a bigger law book
poses "(x + 1) mod N" directly, the learner solves it, but the shortest program for it overwrites its own inputs, so
it cannot be counted with, and the link stays the piece it uses.

**One size up.** With every N up to 26 the same learner again invents it in round 0, with the pairs signal (only
the pair (square, link) makes it short: 150 bits saved, against 23 for the best other pair) and with the hardest-first
rule. There its best wrong attempt is plain x², since the shortcuts that fit small N fit less of the bigger data.

**What it costs.** Choosing is the expensive part. At N ≤ 7, trying all 192 candidates (114 of them with a
million-state search) takes 2.2 search-hours a round, and the pairs signal 3.8 more; our list took 0.56 search-hours
in all. With only the law book and the hardest-first rule it took 0.46. The pairs signal grows with the square of
the pool: with the law book doubled it took 29 search-hours. At N ≤ 26 the learner took 11.3 search-hours with the
hardest-first rule and 24.8 with the pairs signal; our list took 1.57.

**What this says about Bar 3.** The ingredient the inventing test found missing, good practice problems, can come from
the learner itself at this size: from a general law book applied to the data, or just from asking "what is left after
my best wrong attempt, on easy inputs?". What it still took from us is generic but real: the sources themselves (a
law book, a small grammar of restrictions), a search that reaches 6-instruction programs, the rule that routines
leave their inputs alone, and up to 10 times the search of our list to choose (from 0.8 times, with the law book
alone and the hardest-first rule, to 10.6 with every source and the pairs signal; 7 to 16 times at N ≤ 26). It is one tiny
target on a counting machine whose algorithm counts in unary; a general algorithm learner would have to repeat this
on the digit machine and on other targets.
<!-- /PLAIN -->

## Details

### 1. Setup (the inventing study's, unchanged)

- **Machine**: the bare register machine of `library/inventing/brm.py`: registers X = x, N = N, A (the answer), T, U;
  `INC`, `DEC` (stops at 0), `CLR`, `MOV` on every register; `L.r[...]` (repeat r times, r read on entry) and
  `Z.r[...]` (run if r = 0). 35 tokens and 10 loop kinds.
- **Search**: breadth-first by cost, one program per distinct machine state, loop bodies of cost at most 2
  (`library/inventing/bfs.py`), with two additions here (`bar3/search.py`): each level records how many probes its
  best state gets right, and the held levels' best near misses can be read back as programs.
- **Target**: x² mod N on every pair x < N ≤ 7 (28 pairs). A program is accepted when it is right on all of them and
  on every pair with N ≤ 10 (55 pairs, the check size); it is *general* when it is right on every pair with N ≤ 40.
- **Merges and routines**: MDL at half price over the practiced problems' programs, then every practiced problem's
  chosen program becomes a routine `T.<name>` (`wake_sleep.promote`'s rule): the setting with which arm B-bare
  invented x² mod N (INVENTING.md §4.2).

### 2. Candidate practice problems, without curation (`bar3/problems.py`)

Every candidate is a function the machine should compute, A from X and N, posed on instances at the search size
(N ≤ 7) and checked at N ≤ 10. A candidate is kept when it is defined on at least 4 instances at each size, all its
numbers fit the machine (below the cap 2k²), its answers vary, and it is a function of the inputs. Candidates that pose
the same problem are merged (the record keeps the other names under `also`); the target itself is dropped.

- **(a) Laws applied to the data's own numbers** (`laws`, source `law`). A general arithmetic library: succ, pred,
  add, monus (subtraction stopping at 0), absdiff, mul, div, mod, min, max, lt, le, eq, gcd. Each operation is applied
  to the machine's two inputs in every binding ((X, N), (N, X), (X, X), (N, N); X or N for the unary ones), with X
  built from the data's own pairs by the division law's form a + q·b: X = x; X = x + N; X = x + qN with q a coin in
  {0, 1}; X = x + QN with Q another row's x. 114 problems, 68 distinct at N ≤ 7. The library is written about
  operations, never about the target; it contains analogues of most of arm B's curated tasks (add, monus, lt, eq,
  mul, sq, the link = mod for X < 2N, the reduction = mod for X < N²) among some fifty that have nothing to do with it
  (gcd, div, absdiff, max, mod(N, X), ...).
- **(b) Transforms of the target** (`restrictions`, `projections`, `from_near_misses`):
  - *restrict*: the target's pairs where t₁ < t₂, for every ordered pair of terms from 1, 2, 3, X, N, X+X, N+N, X+N,
    X·X, X·N, N·N; one per distinct subset, between 6 pairs and 80% of them (18 at N ≤ 7);
  - *project*: the answer's lowest binary digit, the rest (y div 2), [y = 0], [y < x], [y = x];
  - *the partial computations of the learner's own programs*: the target is probed first (30k states), and its 12
    best near misses (most pairs right, then the cheapest) are run item by item. Each value one of their registers
    holds after an item (not an input, not constant) becomes a problem of its own (source `value`), and the rest of
    the way from it to the answer, y from (X = that value, N), becomes one (source `remainder`) whenever y is a
    function of it; each remainder also comes with its restrictions by the same grammar (source `restrict`, named
    `restrict/remainder[...]`).
- **(c) Dreams** (`dreams`, source `dream`): 16 random programs of the current library, run on the target's inputs
  (arm D's sampler, `library/inventing/loop.py`).

In round 0 at N ≤ 7 the full pool has 192 problems: 68 laws, 18 restrictions of the target, 5 projections, 12 values
and 6 remainders from the 12 near misses, 76 restrictions of those remainders, 7 dreams (after merging duplicates).
The near misses were the target probe's best attempts: `INC.A Z.X[DEC.A]` ("1 if x > 0", right on 18 of 28),
`L.X[L.X[INC.A]]` (x², 16), `MOV.A.X DEC.X L.X[MOV.A.X]` (16), `MOV.A.X` (x, 15), `L.X[DEC.N MOV.A.N]` (15), ...

### 3. How the learner chooses (`bar3/learner.py`)

Each round: probe the target; pose every candidate not yet practiced; probe each (30k states, then 1M states for
those still unsolved: up to cost 6 on the base vocabulary, where the link lives); sort them into *trivial* (solved at
cost ≤ 1: nothing to learn), *learnable* (solved at cost ≥ 2), *duplicate* (learnable, but its program does on random
machines what an earlier learnable one's does) and *unsolved*; pick up to 4 by the rule; practice (a pick the probe
did not solve gets a full 3M-state search); merge (MDL) and make routines; search the target with the full budget
(3M states). Stop when the target's program is general; at most 5 rounds.

| rule | the learner's signal | picks |
|---|---|---|
| `ct2` | target compression, pairs: for every pair of learnable candidates, the target probed (30k states) with both their programs as routines; score = bits saved on the target's description | the best pairs' members, up to 4 |
| `ct1` | the same, one candidate at a time | the 4 best |
| `lp` | learning progress: candidates neither solved before nor hopeless now (learnable at the probe budget), the hardest first | the 4 costliest |
| `nov` | novelty: on the target's pairs, how many answers the candidate's program computes differently from every existing token | the 4 most novel |
| `random_ok` | the probes only | 4 learnable at random |
| `random` | nothing | 4 candidates at random, each given a full practice search |
| `all` | the probes only | every learnable candidate |
| `curated` | arm B's list (`target.CURATED` without the target) | all of it; unsolved tasks searched again each round |

The target's description length under a library (`mdl`): for the best state of each level, its cost in tokens at
log₂(vocabulary) bits each, plus, for every probe it still gets wrong, log₂ 28 + log₂ 7 bits (which probe, and its
answer); a level whose only perfect fits fail the N ≤ 10 check counts one probe wrong. With the base vocabulary it is
88.4 bits (a cost-5 program right on 20 of 28).

### 4. Which of a problem's shortest programs becomes its routine

A solved problem usually has several shortest programs, and only one becomes its routine. The inventing study keeps
the one with the shortest expansion, the first in search order on ties (`wake_sleep.simplest`); call that **v1**. For
the two link candidates the learner posed (the law candidate "X mod N for X = x + qN, q in {0, 1}" and the remainder
of its own near miss x² restricted to X < 2N) there are 6 shortest programs each, all of cost 6, and the first in
search order uses N as its scratch register:

    MOV.A.X L.N[MOV.N.X DEC.X] L.N[MOV.A.X]      (law candidate: right, but N is left at X − N + 1 or 0)
    MOV.A.X L.N[MOV.N.A DEC.A] Z.N[MOV.A.X]      (remainder candidate: N as the borrow)

They compute the link once, but N is gone afterwards, so a loop of them (the reduction) fails. Two of the six leave
the inputs alone and keep the borrow in T or U, `MOV.A.X L.N[MOV.T.A DEC.A] Z.T[MOV.A.X]`: exactly arm B-bare's
`T.csub`. The curated csub task draws its inputs differently, and its first shortest program happens to be the T one.
**v2** ("clean", `choose`) keeps, among the shortest programs, one that leaves its inputs X and N as they were and
changes the fewest registers, then the shortest expansion: the usual convention for a subroutine (no side effects on
its arguments), applied to every problem alike. Runs named `_r1` are v1, `_clean_r2` to `_clean_r5` are v2;
everything else is the same.

### 5. Results at N ≤ 7

Every run, round 0 unless said otherwise (states and search-hours over all of a run's jobs: probes, lookaheads,
practice and target searches; a probe answered from the shared cache still counts what it cost):

| run | sources | rule | routines | x² mod N | states | search-hours: probes / lookahead / practice + target |
|---|---|---|---|---|---|---|
| `curated_clean_r2` | our list | curated | v2 | round 0: `T.sq L.X[MOV.X.A T.csub]` | 91.7M | – / – / 0.56 |
| `curated_r1` | our list | curated | v1 | round 0: the same | 91.8M | – / – / 0.57 |
| `ct2_clean_r2` | all | ct2 | v2 | round 0: `T.mul(X,X) L.X[MOV.X.A T.mod(X,N)@x+qN]` | 586.6M | 2.15 / 3.80 / 0.01 |
| `ct2_law_clean_r4` | laws | ct2 | v2 | round 0: the same | 77.1M | 0.40 / 1.19 / 0.02 |
| `ct2_near_clean_r4` | near misses (value, remainder) | ct2 | v2 | round 0: `T.val1 L.X[MOV.X.A T.res2]` | 421.4M | 1.67 / 0.73 / 0.01 |
| `ct2_nolaw_clean_r4` | all but laws | ct2 | v2 | round 0: `T.res1 L.X[MOV.X.A T.res2]` | 488.7M | 1.76 / 1.50 / 0.01 |
| `lp_clean_r2` | all | lp | v2 | round 0: `L.X[L.X[INC.A]] L.X[MOV.X.A T.mod(X,N)@x+qN]` | 531.5M | 2.15 / – / 0.08 |
| `lp_law_clean_r3` | laws | lp | v2 | round 0: the same | 85.9M | 0.40 / – / 0.06 |
| `lp_near_clean_r3` | near misses | lp | v2 | round 0: `L.X[L.X[INC.A]] L.X[MOV.X.A T.res1]` | 420.0M | 1.67 / – / 0.05 |
| `lp_nolaw_clean_r3` | all but laws | lp | v2 | round 0: `T.res4 L.X[MOV.X.A T.res1]` | 456.7M | 1.76 / – / 0.01 |
| `all_clean_r2` | all | all (46 routines) | v2 | round 0: `T.mul(X,X) L.X[MOV.X.A T.mod(X,N)@x+qN]` | 514.8M | 2.15 / – / 0.08 |
| `ct1_clean_r2` | all | ct1 | v2 | no | 535.8M | 2.15 / 0.14 / 0.05 |
| `nov_clean_r2` | all | nov | v2 | no | 533.3M | 2.15 / – / 0.09 |
| `random_ok_clean_r2`, `_s1`, `_s2` | all | random_ok | v2 | no (3 seeds) | 529–539M | 2.15 / – / 0.05–0.12 |
| `random_clean_r2`, `random_s1/s2_clean_r3` | all | random | v2 | no: 3 seeds × 5 rounds (20 picks each, each given a full search) | 260–284M | – / – / 1.71–2.40 |
| `lp_trans_clean_r3` | restrict, project | lp | v2 | no: nothing learnable but x² where x² < N | 107.2M | 0.30 / – / 0.04 |
| `lp_dream_clean_r3` | dreams | lp | v2 | no | 15.2M | 0.01 / – / 0.04 |
| `ct2_r1` | all | ct2 | v1 | no: no pair makes the target solvable | 654.4M | 2.20 / 6.11 / 0.11 |
| `lp_r1` | all | lp | v1 | no | 528.4M | 2.20 / – / 0.06 |
| `all_r1` | all | all (58 routines) | v1 | no | 518.5M | 2.20 / – / 0.09 |
| `lp_law2_clean_r5` | all + depth-2 laws | lp | v2 | round 1 (§7): `T.res5` = `L.X[L.X[INC.A]] L.X[MOV.X.A T.mod(X,N)@x+qN]` | 1,446.0M | 6.05 / – / 0.06 |
| `ct2_law2_clean_r6` | all + depth-2 laws | ct2 | v2 | round 0: `T.law4 L.N[MOV.X.A T.mod(X,N)@x+qN]` (x² + N, then N links) and `M0.2 L.X[...]` | 1,086.0M | 2.78 / 26.63 / 0.02 |
| `all_law2_clean_r5` | all + depth-2 laws | all (101 problems) | v2 | round 0: `M0.9 L.X[MOV.X.A T.mod(X,N)@x+qN]` and `T.law69 L.N[...]` (§7) | 728.4M | 2.78 / – / 0.67 |

Every program found is general: right on all 820 pairs with N ≤ 40. Expanded, the learner-chosen ones are the
curated one up to the scratch register: `L.X[L.X[INC.A]] L.X[MOV.X.A MOV.A.X L.N[MOV.T.A DEC.A] Z.T[MOV.A.X]]` (or
U for T): square by counting, then x times: copy A to X, and subtract N unless that goes below zero.

**Per rule:**
- **ct2 (pairs).** In every pool that holds a link it ranks exactly the (square, link) pairs first, and only those
  make the target solvable within the 30k-state lookahead: 2 of 1,035 pairs with all sources (both at 66.2 bits saved;
  the best other pair 16.4), 1 of 276 with laws only (66.2 against 12.9), 1 of 190 with near misses only, 1 of 435
  without laws. The square comes from the law book ("X·X"), from the near miss itself ("the value A holds after
  `L.X[L.X[INC.A]]`"), or from a restriction of the target ("x² mod N where x² < N", which is x²). The other picks
  fill the 4 slots from the next pairs and are never used. The lookahead is the expensive part: one small target
  search per pair, 3.8 search-hours for 1,035 pairs (most of it building the search items of each pair's vocabulary).
- **lp (hardest first).** Works whenever a link is posed, because in these pools the link is the costliest learnable
  problem (cost 6; next are min/max/compare at 5) and the full target search then builds the square from primitives
  (cost 3) next to a loop of links (cost 3): a cost-6 program, found while streaming level 6 (17M states). It does
  not need the square as a routine, so it picks it only by accident. When the pool holds more cost-6 problems (§7)
  it needs a second round.
- **all.** Practicing every learnable problem works at this size: 46 problems give 40 routines and 12 merges, a
  vocabulary of 87 tokens (33,247 search items against 5,988 for the base 35), and the target is still at cost 4
  (950k states). It needs the clean routines too (`all_r1`, 58 problems with v1 routines: no).
- **ct1 (single lookahead).** Fails in round 0: alone, the square saves nothing on the target's description (−4.3
  bits: the extra token costs more than it shortens), and both links tie at +5.4 bits with four shortcut helpers
  ("the value A holds after `L.X[DEC.N MOV.A.N]`", "N div X", ...). The top single is a near miss's value, `L.X[MOV.A.X
  MOV.X.N]` (+10.9). It picks the top 4 and misses both links.
- **nov (novelty).** Can never pick the link: on the target's own pairs (x < N), "x mod N for x < 2N" answers exactly
  what "copy x" answers, so its novelty is 0. It picks the problems most unlike anything (N + 1, N + N, N·N, ...).
- **random_ok / random.** No success in round 0 on 3 seeds each, and none for `random` in 5 rounds on 3 seeds (60
  random picks in all, each given a full 3M-state search; none was a link). With 2 links among 44–46 learnable
  problems, 4 random picks contain one with probability 0.17 a round; among all 192 candidates, 0.04.
- **Wrong picks price the right ones out of reach.** After round 0, ct1 and random_ok each hold four irrelevant
  routines (and a few merges). In round 1 the link candidate is no longer learnable at the 1M-state probe: with the
  bigger vocabulary its level 5 grows from 0.46M states to 1.87M (ct1) and 1.51M (random_ok), past the 1.5M that can
  be held, so the probe stops at cost 5 (read from the shared cache, `ct1_clean_r2` and `random_ok_clean_r2`, round 1).
  Neither rule can then pick the link through its probes, so their round 1 was cancelled unfinished (after about 75
  minutes: each probe of a hard candidate also grew from about a minute to the 10-minute limit). `random` gives its
  picks a full 3M-state search, which still reaches cost 6, so it was left to run (§9).

### 6. What it chose, against our list

**The learnable pool already held our list.** Of the 46 learnable problems of round 0 (all sources, v2), 13 have a
routine that solves one of the curated tasks on that task's own instances (`summarize.covers`): together they solve
12 of the 16 (add, dbl, inc, incN, monus, monusN, lt, eq, mul, sq, csub, csub1), and trivial ones solve copy and copyN;
only the two full reductions (x mod N for x < N², the curated `mod` and `modN1`) were out of the probes' reach. The
other 33 learnable problems are noise: the values of shortcut near misses, N div X, N·N, 2N, dreams, ...

**Picks, in order** (round 0; every successful run stopped there):

| run | 1st | 2nd | 3rd | 4th | curated tasks solved by the picks |
|---|---|---|---|---|---|
| ct2, all sources | X·X (law) | X mod N, X = x + qN (law): the link | x² mod N where x² < 2N (its own remainder): the link | N div X, X = x + N | sq, csub, csub1 |
| ct2, laws | X·X | X mod N, X = x + qN | X div N, X = x + qN | N mod X, X = x + N | sq, csub, csub1 |
| ct2, near misses | x² (the value of its near miss `L.X[L.X[INC.A]]`) | the remainder link | the value of `L.X[DEC.N MOV.A.N]` | the value of `L.X[MOV.A.X MOV.X.N]` | sq, csub, csub1 |
| ct2, no laws | x² mod N where x² < N (a restriction of the target: x²) | the remainder link | the two values above | | sq, csub, csub1 |
| lp, all sources | the law link | the remainder link | min(X, N), X = x + qN | max(X, N), X = x + qN | csub, csub1 (+ copy, copyN) |
| all, all sources | every one of the 46 | | | | 14 of 16 (all but mod, modN1) |
| curated (ours) | copy, copyN, inc, incN, add, dbl, monus, monusN, lt, eq, csub, csub1, mul, sq | (mod, modN1: not solved) | | | 14 of 16 |

(Every link also solves `copy`: on x < N it copies x.) The learner keeps the two pieces our skeleton needs, the square
and the link, and none of the other twelve
(addition, comparison, subtraction, multiplication, ...). The curated arm used the same two: its program is
`T.sq L.X[MOV.X.A T.csub]`, and its other 12 routines sit unused. The order is the pairs' order: the square first
because it is in every winning pair.

### 7. The wrap-around concept

The inventing study's missing step was "add one, wrap around at N" (`MI = INC.A DEC.T Z.T[CLR.A MOV.T.N]`): handed
that token, its search found x² mod N in a second. The learner here never invents it. What it uses instead, found
twice from different sources, is the link: x mod N for x < 2N, "subtract N unless that goes below zero", which is
"wrap around once"; repeated x times after squaring it reduces any x² < xN.

- **From the law book**: "(a + q·b) mod b = a mod b" instantiated with the data's x, its N and a coin q ∈ {0, 1}:
  "X mod N for X = x or x + N". The law candidates with Q another row's x (X up to N²: the full reduction) stay
  unsolved at the probe budget; the link is the largest piece of the reduction the probes can reach.
- **From its own attempt**: "x² mod N given x²" (the rest of the way after its near miss x²) is out of reach (75% at
  1M states), but restricted to x² < 2N it is the link (cost 6, 30 s).

**With the wrap step on offer.** The depth-2 law book (`laws2`: every law operation applied to the result of a first
one: `lp_law2_clean_r5`, `all_law2_clean_r5`, `ct2_law2_clean_r6`) poses "(x + 1) mod N" itself. The learner solves
it (cost 5), but its only shortest program is `INC.X L.X[DEC.N] L.N[MOV.A.X]`, which overwrites both X and N: it can be
done once, not counted with. A version that keeps X and N exists one token up, `MOV.A.X INC.A MOV.T.N L.A[DEC.T]
Z.T[CLR.A]` (cost 7, T as scratch), and as a routine it would give another 6-token x² mod N (square, move the count to
U, clear A, then U times: A ← (A + 1) mod N); but a problem keeps only its shortest programs, so that version never
becomes a routine. So even when the step is posed and solved, the reusable wrap piece is the link.

**The doubled pool** (279 candidates, 101 learnable) is also a test of the rules. It slows the hardest-first rule by a
round: with the depth-2 laws there are 13 learnable problems of
cost 6, lp takes the first four (the law link and three depth-2 comparisons), and the larger vocabulary stops the
3M-state target search at cost 5, one short of "x² from primitives, then a loop of links" (`lp_law2_clean_r5`, round
0). In round 1, with the link a routine, the hardest learnable problem is a restriction of the target, "x² mod N where
x > 3", whose shortest program is now the whole algorithm, `L.X[L.X[INC.A]] L.X[MOV.X.A T.mod(X,N)@x+qN]` (on large x
no shortcut fits), and made a routine it *is* the answer: the target at cost 1, general. The same round also turns
the link into the full reduction for another law problem ("N mod X" = `MOV.A.N MOV.N.X L.A[MOV.X.A T.mod(X,N)@x+qN]`).
Two rounds, 6.1 search-hours (1.35 h small probes, 4.70 h big ones: 302 candidates in round 1).

The pairs signal holds on the doubled pool (`ct2_law2_clean_r6`): of 5,050 pairs of its 101 learnable problems,
exactly 4 make the target solvable, all at 66.2 bits saved (the best other pair 35.7): the square or "x·x + N" (a
depth-2 law problem) together with either link. It picks the square, both links and "x·x + N", and finds the target
at cost 4 in round 0. But the price of choosing grows with the square of the pool: 5,151 lookaheads took 26.6
search-hours (29.4 in all), against 3.8 for the 1,081 of the base pool.

Practicing everything still works on the doubled pool (`all_law2_clean_r5`): 101 problems give 117 routines and
merges, a vocabulary of 152 tokens (99,323 search items), and the target search still reaches cost 4 by streaming
37.5M candidates. Besides the square-then-links program it finds an alien variant, `T.law69 L.N[MOV.X.A
T.mod(X,N)@x+qN]`: `T.law69` is the depth-2 law problem "x·x + N", so it starts from N + x² and runs N links (enough,
since N + x² < N·N). Round 0, 3.45 search-hours.

### 8. One size up: every N up to 26

The same configs with k = 26 (351 pairs, 32 probes sampled for the search, checked on every pair with N ≤ 30;
probe B raised to 2M states; `BIG26`). The machine's cap and step limit grow with k, so the counting algorithm still
fits.

- **Curated** (`curated_k26_clean_r3`): round 0, `T.sq L.X[MOV.X.A T.csub]`, general; 31.3M states, 1.57
  search-hours (the searches run slower per state on the bigger numbers).
- **lp** (`lp_k26_clean_r3`, all sources): round 0, `L.X[L.X[INC.A]] L.X[MOV.X.A T.mod(X,N)@x+qN]`, general. The
  pool is 152 problems (67 laws, 58 restrictions, 5 projections, 12 values, 4 remainders, 6 dreams), 43 learnable.
  The near misses change with the size: x² is now the learner's *best* attempt (16 of 32 probes; "1 if x > 0" drops
  to third, 13), since the shortcuts that fit small N fit less of the bigger data. The law link and the remainder link
  are learnable again at cost 6 (449 s and 429 s at the 2M-state probe) and now have the same program (U as the
  borrow), so the second counts as a duplicate. lp picks the link and three cost-5 problems (min, max, N < X), and
  the target search builds x² from primitives next to a loop of links (cost 6, 25.9M states). 423.7M states, 11.3
  search-hours, against 1.57 for the curated list at this size: probing 152 candidates on the bigger numbers took
  10.9 of them (44 of its 71 two-million-state probes stopped at the 10-minute limit).
- **ct2** (`ct2_k26_clean_r7`, all sources): round 0, `T.mul(X,X) L.X[MOV.X.A T.mod(X,N)@x+qN]`, general. Of the 903
  pairs of its 43 learnable problems exactly one makes the target solvable, (square, law link), at 149.5 bits saved
  (more bits than at N ≤ 7: more probes, bigger answers); the best other pair saves 23.4. It picks the square, the
  link, a dream and "N < X". 461.8M states, 24.8 search-hours: the same 10.9 of probes as lp, and 13.9 of lookahead
  (946 small target searches on the bigger numbers).

### 9. How hard each variant was tried

Budgets for every run: probe A 30k states (120 s), probe B 1M states (600 s; 2M at N ≤ 26), lookahead 30k states,
practice and target searches 3M states (2,400 s), loop bodies of cost ≤ 2, at most 5 rounds, 4 picks a round (all
learnable for `all`). Every job's states and seconds are in the records (`summarize.py --effort`); a job answered
from the shared cache counts what it cost when first run.

<!-- EFFORT -->
| run | rounds finished | outcome | probe jobs (a / b) | lookaheads | practice searches | target searches | states | search-hours |
|---|---|---|---|---|---|---|---|---|
| all_clean_r2 | 1 | general in round 0 | 192 / 114 | 0 | 0 | 1 | 514.8M | 2.24 |
| all_law2_clean_r5 | 1 | general in round 0 | 279 / 152 | 0 | 0 | 1 | 728.3M | 3.46 |
| ct1_clean_r2 | 1 | not found | 192 / 114 | 46 | 0 | 1 | 535.8M | 2.35 |
| ct2_clean_r2 | 1 | general in round 0 | 192 / 114 | 1081 | 0 | 1 | 586.6M | 5.96 |
| ct2_k26_clean_r7 | 1 | general in round 0 | 152 / 71 | 946 | 0 | 1 | 461.8M | 24.84 |
| ct2_law2_clean_r6 | 1 | general in round 0 | 279 / 152 | 5151 | 0 | 1 | 1,086.0M | 29.43 |
| ct2_law_clean_r4 | 1 | general in round 0 | 68 / 25 | 300 | 0 | 1 | 77.1M | 1.61 |
| ct2_near_clean_r4 | 1 | general in round 0 | 110 / 79 | 210 | 0 | 1 | 421.4M | 2.42 |
| ct2_nolaw_clean_r4 | 1 | general in round 0 | 132 / 89 | 465 | 0 | 1 | 488.7M | 3.27 |
| curated_clean_r2 | 1 | general in round 0 | 0 / 0 | 0 | 16 | 1 | 91.7M | 0.56 |
| curated_k26_clean_r3 | 1 | general in round 0 | 0 / 0 | 0 | 16 | 1 | 31.3M | 1.57 |
| lp_clean_r2 | 1 | general in round 0 | 192 / 114 | 0 | 0 | 1 | 531.5M | 2.23 |
| lp_dream_clean_r3 | 1 | not found | 16 / 0 | 0 | 0 | 1 | 15.2M | 0.05 |
| lp_k26_clean_r3 | 1 | general in round 0 | 152 / 71 | 0 | 0 | 1 | 423.7M | 11.28 |
| lp_law2_clean_r5 | 2 | general in round 1 | 581 / 295 | 0 | 0 | 2 | 1,445.9M | 6.11 |
| lp_law_clean_r3 | 1 | general in round 0 | 68 / 25 | 0 | 0 | 1 | 85.9M | 0.46 |
| lp_near_clean_r3 | 1 | general in round 0 | 110 / 79 | 0 | 0 | 1 | 420.0M | 1.72 |
| lp_nolaw_clean_r3 | 1 | general in round 0 | 132 / 89 | 0 | 0 | 1 | 456.7M | 1.77 |
| lp_trans_clean_r3 | 1 | not found | 23 / 20 | 0 | 0 | 1 | 107.2M | 0.34 |
| nov_clean_r2 | 1 | not found | 192 / 114 | 0 | 0 | 1 | 533.3M | 2.24 |
| random_clean_r2 | 5 | not found | 0 / 0 | 0 | 20 | 5 | 259.9M | 1.94 |
| random_ok_clean_r2 | 1 | not found | 192 / 114 | 0 | 0 | 1 | 528.5M | 2.26 |
| random_ok_s1_clean_r2 | 1 | not found | 190 / 114 | 0 | 0 | 1 | 537.2M | 2.27 |
| random_ok_s2_clean_r2 | 1 | not found | 191 / 114 | 0 | 0 | 1 | 539.0M | 2.21 |
| random_s1_clean_r3 | 5 | not found | 0 / 0 | 0 | 20 | 5 | 277.7M | 1.71 |
| random_s2_clean_r3 | 5 | not found | 0 / 0 | 0 | 20 | 5 | 284.2M | 2.40 |
| all_r1 | 1 | not found | 192 / 114 | 0 | 0 | 1 | 518.5M | 2.29 |
| ct2_r1 | 1 | not found | 192 / 114 | 1711 | 0 | 1 | 654.4M | 8.42 |
| ct2_trans_r1 | 1 | not found | 23 / 20 | 1 | 0 | 1 | 140.7M | 0.42 |
| curated_r1 | 1 | general in round 0 | 0 / 0 | 0 | 16 | 1 | 91.8M | 0.57 |
| lp_r1 | 1 | not found | 192 / 114 | 0 | 0 | 1 | 528.4M | 2.26 |
| nov_r1 | 1 | not found | 192 / 114 | 0 | 0 | 0 | 513.8M | 2.21 |
| random_ok_r1 | 1 | not found | 192 / 114 | 0 | 0 | 0 | 513.8M | 2.16 |
| random_ok_s1_r1 | 1 | not found | 190 / 114 | 0 | 0 | 1 | 543.2M | 2.29 |
| random_ok_s2_r1 | 1 | not found | 191 / 114 | 0 | 0 | 1 | 537.5M | 2.28 |
| random_r1 | 2 | not found | 0 / 0 | 0 | 8 | 2 | 80.7M | 0.29 |
| random_s1_r1 | 1 | not found | 0 / 0 | 0 | 4 | 1 | 60.9M | 0.18 |
| random_s2_r1 | 2 | not found | 0 / 0 | 0 | 8 | 2 | 129.6M | 0.28 |
<!-- /EFFORT -->

Stopped early, and why (v1 was superseded by v2 once the side-effect problem was found, and the workspace was
over Modal's 100-container cap):
- v1: `curated_r1` finished (round 0). `ct2_r1`, `lp_r1` and `all_r1` finished round 0 (not found), which settles the
  v1 question, and were cancelled in round 1; `random_r1` after 2 rounds, `random_s2_r1` after 2, `random_s1_r1`
  after 1, `random_ok_r1` / `_s1` / `_s2` after round 0. `nov_r1` and `random_ok_r1` lost their round-0 target search
  to the cancellation. `ct1_r1` and `ct2_{law,nolaw,near,dream}_r1` were cancelled before finishing round 0 (no
  record). `ct2_trans_r1` finished round 0 with no pick: with a single learnable candidate the pair rule had no pair
  (fixed afterwards: it falls back to the single scores).
- `nov_clean_r2`, `lp_trans_clean_r3`, `lp_dream_clean_r3`: cancelled after round 0; their failure is structural
  (novelty scores the link 0 on the target's pairs; the other two pools pose no link at all).
- `random_ok_s1_clean_r2`, `random_ok_s2_clean_r2`: cancelled in round 1 to free containers (seed 0 continued).
- `ct1_clean_r2`, `random_ok_clean_r2`: cancelled in round 1 after about 75 minutes, when the cache showed the link no
  longer learnable at their round-1 probe budget (§5).
- `random_clean_r2`, `random_s1_clean_r3`, `random_s2_clean_r3` ran all 5 rounds.

### 10. Files and how to rerun

Everything is in `cpu_first/bar3/` (nothing outside it changed; it imports `library/` and `library/inventing/`):
- `problems.py`: the candidates (laws, depth-2 laws, restrictions, projections, values and remainders of near misses,
  dreams, the curated list, the target).
- `search.py`: the inventing study's breadth-first search plus per-level best fits and near misses.
- `jobs.py`: one search job (probe, practice, target, lookahead), with acceptance at N ≤ 7 and N ≤ 10.
- `learner.py`: the loop, the rules, the routine choice (`choose`, `promote_first`) and every config (`CONFIGS`).
- `modal_practice.py`: Modal (one non-preemptible driver per config, searches fanned out to CPU containers, identical
  jobs shared through a `modal.Dict` with in-flight claims). Not the shared runner (`tools/modal_runner.py`): that
  one runs submission files through the official evaluator, and these are search jobs fanned out by a driver.
  Limits: every search has its own state and time budget, containers have timeouts and hard memory limits (added
  after the runs, which had 16 GiB requests and no limit), and a cancelled driver's inputs are cancelled with it.
  The shared cache is the one deliberate cache: it let 40-odd arms share their identical round-0 probes. `summarize.py`, `inspect_round.py`: tables and rounds.
- The shared cache keys every search job by its content, so a rerun of an unchanged config returns the stored
  records in minutes; bump `VERSION` in `modal_practice.py` to search again from scratch.
- `records/<config>_<tag>.jsonl`: one line per round (pool, every probe's status, scores, pairs, picks, merges,
  routines, the target probe and attempt, running search totals); `calls_<tag>.json`: the Modal call ids;
  `tiny_local.jsonl`, `tiny_smoke.jsonl`: the smoke test (k = 5, tiny budgets), local and on Modal.

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.learner --config tiny
```

```bash
cd cpu_first && BAR3_MAX=16 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/modal_practice.py --configs ct2_near_clean,lp_law_clean --tag rerun
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/modal_practice.py --fetch
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.summarize --table bar3/records/*_clean_r*.jsonl
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.inspect_round bar3/records/ct2_clean_r2.jsonl 0
```
