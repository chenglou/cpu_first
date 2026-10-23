# Breadth: does the self-practicing learner invent other formulas, unchanged?

*Written 2026-10-23. Code in `cpu_first/bar3/breadth/`, records in `bar3/breadth/records/`. Nothing committed, nothing
submitted, no file outside `bar3/breadth/` and this write-up changed.*

## In plain words

<!-- PLAIN -->
**The question.** In [PRACTICE.md](PRACTICE.md) a learner searched for short programs on a bare counting machine. It
picked its own practice problems and turned their solutions into new instructions, and in its first round it
invented x² mod N (square x, keep the remainder after dividing by N) at a tiny size, every pair x < N ≤ 7. Was that
about x² mod N, or would the same learner, changed in nothing, invent other formulas too? Here it gets nine more,
one at a time, at the same tiny size:
- seven relatives of x² mod N (the forms of the benchmark's litmus and held-out tasks): x² + 1, x² − 1, x² + 3,
  10·x², 10·x² + 1, x³ (one round of x^(3^T)), and Montgomery's x²·2⁻¹ (with N odd);
- two outsiders: 2x + 1 (one step of a linear congruential generator) and the Fibonacci numbers.

Each is taken mod N. Nothing about the learner changed: the same sources of practice problems (a general law book of
arithmetic, changes of the target, the learner's own best wrong attempts, random programs), the same ways of
choosing, the same rule that routines leave their inputs alone. We also reran x² mod N through the same harness: it
reproduces PRACTICE.md exactly.

**The answer: four of the nine, all in its first or second round.** x² − 1 and 2x + 1 in round 0 with every choosing
rule; x² + 1 in round 0 with the pairs signal (round 1 with "hardest first"); x² + 3 in round 1 with both.
Each time the step the formula needed came from the learner's own sources, not from a list of ours:
- for x² − 1, the target itself on easy inputs ("only where x² < N"), whose program is "square, take N if that is
  0, subtract 1";
- for x² + 1, the number one of its best wrong attempts computes ("x², plus one");
- for 2x + 1, "the rest of the way" after its wrong attempt "double x": add one, then subtract N unless that goes
  below zero. That is the "add one and wrap around at N" step PRACTICE.md never saw;
- for x² + 3, the target at x = 0, "3 mod N", whose program ("add 3, then reduce") works on any number;
- in all four, the link from the law book: "subtract N unless that goes below zero".

**What it did not invent from scratch, and why.** 10·x², 10·x² + 1, x³, Montgomery's x²·2⁻¹ and Fibonacci: not with
any rule in 2 to 5 rounds, and not with our own curated practice list either (two of them came later, with a carried
library).
- The first four need programs of 13 to 22 instructions even when written by hand from the pieces the learner has,
  against the 6 or 7 its searches reach:
  - 10·x² needs the number 10, which a counting machine has to count up to;
  - x³ needs x again after the reduction has used it up, and a second reduction;
  - Montgomery's needs a halving loop.
- Fibonacci needs only 8 instructions, but its repeated step moves two numbers at once. The learner's sources only
  pose problems with one answer computed from x and N, so none of them poses that step.
- At this size the hard targets also look simpler than they are: x³ equals x on 21 of the 28 pairs, so its best
  wrong attempt is "copy x", which poses nothing useful.

**What carried over.** We also ran the targets in sequence, the library persisting and each solved target kept as a
routine ("hardest first"). x² + 1, x² − 1, x² + 3 and 2x + 1 then all came in round 0 (from scratch the same rule
had needed round 1 for x² + 1 and x² + 3). The programs were 1 to 3 instructions over earlier answers:
- x² + 1: "x² mod N, then add one and wrap once";
- x² − 1: "x² + 1 mod N, then subtract two and wrap";
- x² + 3: "x² mod N, then add three and wrap".

The wrap-around steps ("add three and wrap at N", "subtract two and wrap") now appear as routines of their own, each
posed by the learner's wrong attempts at the new target. The concept PRACTICE.md found missing appears as soon as a
formula needs a shift after the reduction.

The carried answers also opened two of the five hard targets, run from the library left after 2x + 1:
- x³ mod N in round 0, by `MOV.U.X L.U[MOV.X.U T.tgt_h1]`: save x, then x times add x² and reduce. The carried x²
  mod N routine adds onto what A holds, so repeating it multiplies by x mod N.
- Montgomery's x²·2⁻¹ mod N in round 0, by an odd 42-instruction program over carried routines: x² mod N, then N
  steps of a routine practiced for x² + 1. It is right on every odd N up to 101.
- In both, the piece the learner practiced was the target itself on most inputs ("only where x > 1"), short over
  the carried routines and nearly the whole answer.
- 10·x², 10·x² + 1 and Fibonacci stayed out of reach. With 34 to 41 routines the target search only got to
  programs of cost 4, against 6 from scratch.

But invention did not get cheaper in search time. Each round still tries every candidate problem, and the growing
library makes each try slower. x² + 1 fell from 8.6 to 1.9 search-hours (from scratch it took two rounds), but x² −
1, x² + 3 and 2x + 1 rose from 2.3 to 6.0, 6.3 to 9.7 and 2.3 to 6.9. A hard target's round with the 34-routine
library cost 9 to 28 search-hours.

**The cost of choosing.** The pairs signal ("which two problems, made into instructions, make the target shortest")
tries every pair of the problems it can solve.
- That is 630 to 1,953 pairs a round here: 2.7 to 8.1 search-hours, against 1 to 3.6 for trying the candidates.
- It grows faster in later rounds, as each lookahead slows with the bigger vocabulary. Round 1 of x² + 3 had 1,431
  pairs at 33 seconds each instead of 14, so 13.8 hours.
- It grows faster still with a carried library: 89 to 111 learnable problems a step, 3,900 to 6,100 pairs. The pairs
  signal's own sequence ran 5 hours on x² + 1 without finishing and was stopped.

A cheaper rule keeps the same choices: only the pairs with one member among the two hardest problems it can solve.
That is about three lookaheads per problem instead of half their square. Run on all ten targets, it invents the same
four in round 0 with the same two pieces each, choosing with 6 to 14 times less search (0.4 to 0.6 hours instead of
2.7 to 8.1). Ranking problems one at a time does not work: each piece of a winning pair is useless alone. The square
alone, or "x², plus one" alone, shortens nothing.

**What it says about Bar 3.** The self-practicing learner is not a one-formula trick.
- Unchanged, it invents four of the nine new formulas from scratch, each time posing the step the formula needs from
  its own generic sources.
- With its solved formulas kept as routines it invents two more, x³ and Montgomery's, which neither it nor our
  curated list reached from scratch. The new programs are built on the old answers.
- Its limits are where its search reaches (about 6 instructions over what it already has) and the shape of its
  sources (one answer from x and N). A formula that needs a constant (10·x²) or a step that moves two numbers at once
  (Fibonacci) stays out of reach.
- A growing library makes every round dearer, because choosing still tries every candidate.

A general algorithm learner would need:
- practice problems with more than one input and output;
- a way to come by constants;
- choosing that does not cost a search per candidate.
<!-- /PLAIN -->

## Details

### 1. Setup: the learner unchanged, nine new targets

- **The learner.** `bar3/learner.py` as committed (5f39a31), with PRACTICE.md's configs:
  - `lp_clean`: hardest solvable problems first;
  - `ct2_clean`: the pairs signal;
  - references: `all_once`, practice every learnable problem (`all_clean` for one round: its round 1 would probe
    about 150 new candidates with a 90-token vocabulary, some 25 search-hours a target), and `curated_clean`, our 16
    practice problems for x² mod N (arm B's list).

  Every source (law book, restrictions and projections of the target, values and remainders of its own near misses,
  dreams), routines without side effects (v2), 4 picks a round, at most 5 rounds, every budget as in PRACTICE.md §9.
- **The target is the only thing that changes.** The learner reads it from `library/inventing/target.py` (`pairs`:
  the instances; `answers`: the formula) in two places:
  - `bar3/problems.py`: the target's data, the law book's data, dreams;
  - `library/inventing/jobs.py`: `general`, the N ≤ 40 check.

  `breadth/targets.py` points those two functions at the calling thread's target, in this one place. A search job
  that needs the target (a target search's N ≤ 40 check) carries its name, and each worker sets it first. No learner
  file changed.
- **The gate.** x² mod N through the same harness reproduces PRACTICE.md field for field. `lp_clean_h1_b1` matches
  `lp_clean_r2` on the 192 candidates, every probe's outcome, the picks, the library, the target's programs and the
  same 531,512,300 states. `ct2_clean_h1_b1` matches `ct2_clean_r2` the same way, and `all_once_h1_b1` matches
  `all_clean_r2`'s outcome.
- **Size.** PRACTICE.md's main size: every pair x < N ≤ 7 to search on (28 pairs), N ≤ 10 to accept (55), N ≤ 40 to
  call a program general (820). Montgomery's target is defined on odd N only (16, 25 and 400 pairs).
- **Machine.** The bare counter machine of INVENTING.md: registers X = x, N, A (the answer), T and U (scratch); add
  one, subtract one, clear, copy, "repeat r times", "if r = 0". Values saturate at 2k² (98 at N ≤ 7, 200 at N ≤ 10),
  so a formula whose plain value passes that (10·x², x³) has to be reduced along the way.

| name | formula, one round | why it is here | pairs (search / check) | shortest program we could write from the square and the link |
|---|---|---|---|---|
| h1 | x² mod N | the original: the gate | 28 / 55 | 4: `T.sq L.X[MOV.X.A T.link]` |
| rho | (x² + 1) mod N | Pollard's rho map: +1 before the reduction | 28 / 55 | 5: `T.sq INC.A L.N[MOV.X.A T.link]` |
| minus1 | (x² − 1) mod N | −1 is +(N − 1): a wrap below zero | 28 / 55 | 7: `T.sq L.N[INC.A] DEC.A L.N[MOV.X.A T.link]` |
| plus3 | (x² + 3) mod N | +3 needs up to three wraps when N ≤ 3 | 28 / 55 | 7: `T.sq INC.A INC.A INC.A L.A[MOV.X.A T.link]` |
| ten | 10·x² mod N | a constant (10) the machine has to count up to | 28 / 55 | 20 |
| tenrho | (10·x² + 1) mod N | both changes | 28 / 55 | 21 |
| cube | x³ mod N (one round of x^(3^T)) | a reduction in the middle (x³ passes the cap), and x needed after it | 28 / 55 | 13: `MOV.U.X T.sq L.X[MOV.X.A T.link] MOV.T.A CLR.A L.U[L.T[INC.A]] L.U[MOV.X.A T.link]` |
| mont | x²·2⁻¹ mod N, N odd | Montgomery's squaring with B = 2: halve, adding N first when odd | 16 / 25 | 22 |
| affine | (2x + 1) mod N | outside: one step of a linear congruential map | 28 / 55 | 6: `MOV.A.X L.X[INC.A] INC.A MOV.X.A T.link` |
| fib | F_x mod N | outside: a two-state recurrence, (a, b) → (b, a + b) x times | 28 / 55 | 8: `INC.A L.X[MOV.X.A L.U[INC.X] MOV.U.A T.link] MOV.A.U`; its loop body costs 5, the search's at most 2 |

`T.sq` is `L.X[L.X[INC.A]]` and `T.link` is `MOV.A.X L.N[MOV.T.A DEC.A] Z.T[MOV.A.X]`: the learner's own two pieces
for x² mod N. Each program was checked on all three sizes. The searches here reach programs of cost 6, sometimes 7.

At this size several targets nearly coincide with something simpler:
- x³ is x on 21 of the 28 pairs (x³ ≡ x mod 2, 3 and 6);
- 10·x² is 0 on 14 (N = 1, 2, 5);
- F_x is x² mod N on 16.

The learner's near misses find those shortcuts first.

### 2. Results per target, each from scratch

Each cell: the round a general program was found in, or "no" with the rounds run, and the run's search-hours
(probes, lookaheads, practice and target searches; a probe shared through the cache counts in every run that used
it).
- Hardest first ran its full 5 rounds.
- The pairs signal was stopped after 2 rounds wherever round 0 found no pair that solves the target: its round 1
  cost 6.0 to 13.8 lookahead-hours (21 to 33 seconds a lookahead, against 11 to 15 in round 0).
- The cheaper pairs signal and practice-everything ran round 0 only.
- Our curated list on x² mod N is PRACTICE.md's `curated_clean_r2` (round 0, 0.56 h).

| target | hardest first | pairs signal | cheaper pairs (round 0 only) | practice everything (one round) | our curated list |
|---|---|---|---|---|---|
| h1: x² mod N | **round 0** (2.4 h) | **round 0** (5.8 h) | **round 0** (2.7 h) | **round 0** (2.4 h) | – |
| rho: (x² + 1) mod N | **round 1** (8.6 h) | **round 0** (11.4 h) | **round 0** (4.2 h) | **round 0** (3.7 h) | **round 0** (0.5 h) |
| minus1: (x² − 1) mod N | **round 0** (2.3 h) | **round 0** (5.3 h) | **round 0** (2.8 h) | **round 0** (2.4 h) | **round 0** (0.7 h) |
| plus3: (x² + 3) mod N | **round 1** (6.3 h) | **round 1** (22.2 h) | no, 1 round (2.5 h) | no, 1 round (2.3 h) | **round 1** (1.6 h) |
| ten: 10·x² mod N | no, 5 rounds (46.7 h) | no, 2 rounds (15.7 h) | no, 1 round (1.4 h) | no, 1 round (1.3 h) | no, 5 rounds (3.6 h) |
| tenrho: (10·x² + 1) mod N | no, 4 rounds (25.6 h) | no, 2 rounds (14.6 h) | no, 1 round (1.5 h) | no, 1 round (1.5 h) | no, 5 rounds (3.5 h) |
| cube: x³ mod N | no, 5 rounds (38.3 h) | no, 2 rounds (19.4 h) | no, 1 round (2.9 h) | no, 1 round (2.8 h) | no, 5 rounds (3.7 h) |
| mont: x²·2⁻¹ mod N, N odd | no, 5 rounds (27.8 h) | no, 2 rounds (15.4 h) | no, 1 round (2.5 h) | no, 1 round (2.4 h) | no, 5 rounds (1.1 h) |
| affine: (2x + 1) mod N | **round 0** (2.3 h) | **round 0** (10.4 h) | **round 0** (2.9 h) | **round 0** (2.4 h) | **round 0** (0.5 h) |
| fib: F_x mod N | no, 5 rounds (31.2 h) | no, 2 rounds (17.4 h) | no, 1 round (3.1 h) | no, 1 round (2.9 h) | no, 5 rounds (3.4 h) |

Every program found is general (right on all pairs with N ≤ 40). The round-0 numbers of the four arms that share
probes are identical up to the choice: one pool, one set of probe outcomes per target.

**The invented four, and how.** Programs as the learner wrote them, with its routines; every one checked general.

- **x² − 1** (`minus1`): round 0 with every rule.
  - Pieces: the law book's link and "the target where x² < N", a restriction of the target from its small grammar.
  - On those pairs x² − 1 needs no reduction, except at x = 0, where it is N − 1. Its shortest program,
    `L.X[L.X[INC.A]] Z.X[MOV.A.N] DEC.A` (cost 6: "square; if x is 0, take N; subtract one"), is the whole first half
    of the answer.
  - Program (pairs signal): `M0.2 L.N[MOV.X.A T.mod(X,N)@x+qN]` with `M0.2` = `L.X[L.X[INC.A]] Z.A[MOV.A.N] DEC.A`,
    i.e. x² − 1, borrowing N when x² = 0, then N links. Hardest first finds the same, with the merge named `M0.3`.
    Our curated list finds `MOV.A.N T.sq DEC.A L.N[MOV.X.A T.csub]` (start from N − 1, add x², reduce).
  - Pairs signal: of 820 pairs exactly one makes the target solvable in the lookahead, (link, restriction), at 70.4
    bits saved; the next saves 32.3.
- **2x + 1** (`affine`): round 0 with every rule.
  - The near miss `L.X[INC.X] MOV.A.X INC.A` (2x + 1, unreduced) holds 2x in X after its first item.
  - "The rest of the way" from that value, restricted by the grammar (1 < X, N < X + X, ...), is "add one and wrap":
    `INC.X MOV.A.X L.N[MOV.T.A DEC.A] Z.T[MOV.A.X]` (cost 7). The learner posed and practiced the wrap-around step
    itself.
  - Hardest first: `L.X[T.res1] T.res3`, which counts 2x into X and then adds one and wraps (the simplest of its
    programs is `L.X[INC.X] T.res1`).
  - The pairs signal and the cheaper one: `M0.2 T.res3`.
  - 300 of the 1,953 pairs solve it, because the wrap piece already does almost everything.
- **x² + 1** (`rho`): round 0 with the pairs signal and with practice-everything; round 1 with hardest first.
  - Pieces: the link and the value one of its near misses computes, "x², plus one" (`L.X[L.X[INC.A]] INC.A`, right
    on 13 of the 28 pairs). Program `M0.3 L.N[MOV.X.A T.mod(X,N)@x+qN]`, `M0.3` = `INC.A L.X[L.X[INC.A]]`: one,
    plus x², then N links.
  - Pairs signal: of 1,770 pairs exactly one solves it, (link, "x² + 1"), at 91.1 bits saved; the next saves 33.7.
  - Hardest first spends its round 0 on the three cost-6 problems: the link and two shortcut remainders. Without x² as
    a routine, "x², +1, then N links" is cost 7, one more than its target search reaches. In round 1 it finds an odd
    program, general all the same: `M1.2 MOV.N.A T.min(X,N)@x+qN M1.3`, 4 tokens and 27 primitive instructions. It
    computes x² + 1, one link with N as scratch, then the law-book routine "N mod X" with the roles swapped.
- **x² + 3** (`plus3`): round 1 with hardest first, the pairs signal and our curated list; not in round 0 with any
  rule.
  - Round 0 poses nothing that makes it short: x² + 3 needs up to three wraps when N ≤ 3.
  - In round 1, with the link a routine, hardest first picks "the target at x = 0", which is "3 mod N", and solves
    it by `INC.A INC.A INC.A L.A[MOV.X.A T.mod(X,N)@x+qN]`: add 3 to A, then reduce A times.
  - That program does not read X, so after a square it computes x² + 3 mod N. Program: `L.X[L.X[INC.A]] T.res7`.
  - The pairs signal gets it in round 1 too, from the same piece. Of 1,431 pairs, 55 make it solvable; the best is
    (X·X, "the target at x = 0") at 125.5 bits saved. Program `T.mul(X,X) M1.1`, the same algorithm, x² then "+3
    and reduce". Its lookaheads cost 13.8 search-hours that round (about 33 seconds each with the round-0 routines in
    the vocabulary).

**The five it did not invent.** The table's searches got nowhere near:
- the target search's best attempt stayed at 18–22 of 28 pairs for 10·x² and 18–21 for 10·x² + 1;
- 22–25 for x³, most of them from "copy x";
- 11–14 for Montgomery's;
- 27–28 for Fibonacci. Its 28-of-28 attempts from round 2 on were small-N shortcuts (one uses the law book's
  gcd) that fail the N ≤ 10 check.

On these five, from round 1 on, with the first round's routines in the vocabulary, every hardest-first target
search stopped at cost 5, against 6 in round 0.

What each would need:
- **10·x², 10·x² + 1**: the constant 10. The machine has no constants and counts in unary. The law book applies
  operations to x and N only, the target's restrictions use 1, 2 and 3 only in comparisons, and no near miss reaches
  10·x². Its best attempts are shortcuts for small N (10·x² mod N is 0 for N = 1, 2, 5, and x² mod N for N = 3).
- **x³**: x² mod N times x, reduced again. The reduction loop overwrites X, so x has to be saved first, and the
  product passes the machine's cap unless reduced twice: 13 instructions. At N ≤ 7, x³ = x on 21 pairs. The best
  near miss is `MOV.A.X` (copy x), whose values and remainders are the inputs and the target themselves, so it poses
  nothing.
- **Montgomery's x²·2⁻¹**: x² mod N, then halve, adding N first when odd. A halving loop needs a parity toggle inside
  a loop body of cost 5 or more: 22 instructions.
- **Fibonacci**: `INC.A L.X[MOV.X.A L.U[INC.X] MOV.U.A T.link] MOV.A.U` is 8 instructions with the link. Its loop
  body, "(a, b) → (b, a + b mod N)", costs 5, beyond the search's limit of 2 for loop bodies, unless it is a routine.
  As a practice problem it has two inputs (a, b) and two outputs. Every source here poses one answer A from (x, N),
  so none can pose it.

Our curated list fails all five too, after 5 rounds: the misses are about what can be reached and posed, not
about the choice.

### 3. What it chose to practice

**When it invents the target**, its four picks hold the two pieces the program uses. Pieces, by source:

| target | rule, round | the pieces the program uses (source) | the other picks |
|---|---|---|---|
| h1 | pairs, 0 | X·X (law); the link X mod N for X = x + qN (law) | the link again (its own remainder after x²); N div X |
| h1 | hardest, 0 | the link (law); the square is built in the target search | the link again (remainder); min, max |
| minus1 | pairs, 0 / hardest, 0 | the link (law); "target where x² < N" (restriction of the target) | N div X and a near-miss value / min, max |
| rho | pairs, 0 | the link (law); "x², plus one" (value of its own near miss) | N div X; a near-miss value |
| rho | hardest, 1 | the link (round 0); "N mod X" (law, round 1), with roles swapped | three restricted remainders |
| affine | hardest, 0 / pairs, 0 | "add one and wrap" (restricted remainder of its own near miss "2x + 1") | the link and other restrictions of the same remainder |
| plus3 | hardest, 1 | the link (round 0); "target where x < 1" = 3 mod N (restriction of the target) | a restricted remainder; N div X; "target where x < 2" |
| plus3 | pairs, 1 | X·X (law); "target where x < 1" (restriction of the target) | "target where x < 2"; N ∸ X |

Our curated list for x² mod N (16 problems: copy, add, compare, the link, the reduction, the square, ...) solves the
same four, and x² + 3 in round 1 like hardest first. Of its 16 routines, each found program uses 2: the square or
the doubling, and the link or the reduction. So for these targets, too, the learner's own choice keeps what the
answer needs and leaves the rest. The two sources PRACTICE.md found essential were each essential somewhere:
- the target's own restrictions, for x² − 1 and x² + 3;
- its near misses' values and remainders, for x² + 1 and 2x + 1.

**When nothing it can learn makes the target shorter**, both rules fall back on the law book's order, the same picks
for every target:
- *Hardest first* works through the law book by cost:
  - round 0: the link, min, max, [N < X];
  - round 1: N div X, N mod X, |X − N|, [X < N];
  - round 2: the next ones down.

  It deviates only where the target's own transforms pose a problem of cost 6: "target: [y = x]" for Montgomery's,
  near-miss remainders for Fibonacci.
- *The pairs signal* finds every pair's gain equal, about 0 bits, or driven by one useful single. It picks in pool
  order: X + N, X + X, N + N, N ∸ X in round 0, X ∸ N, X·N, X·X in round 1.

So with no pair that helps, the pairs signal carries no information, and neither rule notices that it should look
for problems of another kind (a constant, a second input).

### 4. Carry-over: the targets in sequence

**Protocol** (`breadth/carry.py`). The code that runs is learner.run's own source with three lines patched: start
from a carried state (library, loop kinds, corpus, practiced problems, routine names), and return it at the end.
Between targets the harness does two things:
- it renames the finished target's routine-name keys, so a later candidate with the same name (the same near-miss
  program on another target's answers) gets a fresh routine;
- each solved target joins as a practiced problem: its general programs go into the corpus, and the one `choose`
  keeps becomes the routine `T.tgt_<name>`, as for any practiced problem.

Everything else is the unchanged learner, rule `lp_clean`, in the order x² mod N, x² + 1, x² − 1, x² + 3, 2x + 1,
then the hard five. The hard five were run as *branches*: each alone from the library left after 2x + 1, round 0
only. Their probes, against a 34-token library, took about 2 hours of wall-clock a round. The serial chain also
went on to 10·x².

| step | from scratch (hardest first) | in sequence | program in sequence | uses | library before → after | learnable | search-hours, from scratch → in sequence |
|---|---|---|---|---|---|---|---|
| h1 | round 0 | round 0 | `L.X[L.X[INC.A]] L.X[MOV.X.A T.mod(X,N)@x+qN]` | – | 0 → 7 | 46 | 2.38 → 2.38 |
| rho | round 1 | round 0 | `T.tgt_h1 INC.X T.mod(X,N)@x+qN` | h1's answer, the link | 7 → 14 | 89 | 8.60 → 1.85 |
| minus1 | round 0 | round 0 | `T.tgt_rho T.res10` | rho's answer; "(X − 2) mod N" | 14 → 19 | 97 | 2.29 → 6.04 |
| plus3 | round 1 | round 0 | `T.tgt_h1 M0.7` | h1's answer; "(X + 3) mod N" | 19 → 27 | 111 | 6.29 → 9.73 |
| affine | round 0 | round 0 | `T.res20` | "target where 1 < x" (its program is the whole answer) | 27 → 34 | 89 | 2.32 → 6.88 |
| cube (branch) | no in 5 rounds | **round 0** | `M0.13` = `MOV.U.X L.U[MOV.X.U T.tgt_h1]` | h1's answer, x times | 34 → 41 | 60 | 38.33 (no) → 15.09 |
| mont (branch) | no in 5 rounds | **round 0** | `M0.16 Z.T[M0.8]`, `M0.16` = `T.tgt_h1 L.N[MOV.X.T T.res7]` | h1's answer; `T.res7`, practiced for x² + 1 | 34 → 40 | 78 | 27.76 (no) → 12.05 |
| fib (branch) | no in 5 rounds | no (round 0 only) | – | – | 34 → 40 | 59 | 31.24 (no) → 9.46 (no) |
| tenrho (branch) | no in 4 rounds | no (round 0 only) | – | – | 34 → 39 | 76 | 25.63 (no, 4 rounds) → 28.30 (no) |
| ten (the chain, after 2x + 1) | no in 5 rounds | no (round 0, then stopped) | – | – | 34 → 41 | 72 | 46.71 (no) → 25.15 (no) |

What carried over:
- **Earlier answers, more than the practice routines.** Each later program is built on a solved target: "x² mod N,
  add one and wrap once" for x² + 1; "x² + 1 mod N, then (X − 2) mod N" for x² − 1; "x² mod N, then (X + 3) mod N"
  for x² + 3. In three of the five steps the target probe (30,000 states, before any practice) already solved the
  target with the carried answers, by 3- and 4-token programs such as `MOV.A.N DEC.A T.tgt_h1` (start A at N − 1,
  then let the x² mod N routine add x² and reduce: x² − 1 mod N).
- **Two hard targets, opened by an old answer.** x³ mod N and Montgomery's x²·2⁻¹ mod N were invented in round 0 of
  their branches. Neither was invented from scratch, by any rule, in up to 5 rounds, nor by our curated list.
  - The carried x² mod N routine is `L.X[L.X[INC.A]] L.X[MOV.X.A T.mod(X,N)@x+qN]`: it adds x² onto whatever A
    holds, then reduces. So "save x, then x times: x² mod N" is x·x² mod N. That is the learner's x³ program,
    `MOV.U.X L.U[MOV.X.U T.tgt_h1]`, right on every pair with N ≤ 81 (we checked N = 41 to 81 beyond its own N ≤ 40).
  - Montgomery's, `M0.16 Z.T[M0.8]` with `M0.16` = `T.tgt_h1 L.N[MOV.X.T T.res7]`, is an odd program. It takes x² mod
    N, then N steps of `T.res7` (practiced for x² + 1: "if X is 0 take N, subtract one, add one, then add X² and
    reduce"), fed with the link's leftover borrow; a last correction follows. 42 primitive instructions, right on
    every odd N up to 101 (we checked 41 to 101).
  - In both, the practiced piece was the target itself on most inputs ("only where x > 1"), cost 4 over the carried
    routines. Its program already was the algorithm: `MOV.U.X L.X[T.tgt_h1 MOV.X.U]` for x³.
  - The other three stayed out of reach. 10·x² and 10·x² + 1 still need the constant 10, and Fibonacci still needs
    its two-number step. With 39 to 41 routines their target searches only reached cost 4, against 6 from scratch.
- **The link** is in every program, but carrying it saves nothing: from scratch hardest first picks the law book's
  link in round 0 for every target anyway (it is the costliest learnable problem).
- **Wrap-around steps.** Fresh, the learner posed "add one and wrap" for 2x + 1, and "add 3, reduce" (3 mod N, which
  works on any number) for x² + 3. In the sequence it poses the rest of the way from a carried answer: "(X − 2) mod
  N" from x² + 1 mod N to x² − 1, and the step from x² mod N to x² + 3. Our concept check (each routine run on every
  X < N with N ≤ 10, from X or on A) finds routines computing (X + 1), (X + 3) and (X − 2) mod N. So
  PRACTICE.md's missing concept, "add one and wrap around at N", appears as soon as a formula needs a shift after
  the reduction: from the target's own near misses, without the law book posing it.
- **Not search.** Rounds fell (x² + 1 and x² + 3 in round 0 instead of round 1), but each round still probes every
  candidate (162 to 299 here). Each probe gets slower as the library grows (7 routines to 34), and the target
  search runs with a vocabulary of up to 69 tokens. Search-hours: x² + 1 fell from 8.6 to 1.9, because from scratch
  it needed two rounds; x² − 1, x² + 3 and 2x + 1 rose from 2.3 to 6.0, 6.3 to 9.7 and 2.3 to 6.9. Choosing does
  not know that the target probe has already found the answer.
- **The pool grows with the library**: 89 to 111 learnable problems a step, against 41 to 63 from scratch. For the
  pairs signal that is 3,900 to 6,100 pairs a round. The pairs-signal sequence (`seq_ct2_clean_targets`) finished
  x² mod N in round 0, like the fresh run. It then spent more than 5 hours on 12 containers (about 60
  container-hours) choosing for x² + 1, and was stopped unfinished to free containers. Its library then had 6
  routines; the hardest-first sequence's pool at the same step had 89 learnable problems.

### 5. The cost of choosing, and a cheaper pairs signal

**How the pairs signal's cost grows.** For L learnable problems the pairs signal runs L + L(L − 1)/2 lookaheads (a
30,000-state search of the target with the candidate routines added). Each costs about 11 to 15 seconds with the
base vocabulary, mostly building the search items of that vocabulary, and more as the library grows.

| run | learnable L | lookaheads | seconds each | lookahead search-hours | probe search-hours, same round |
|---|---|---|---|---|---|
| ten, round 0 | 36 | 666 | 14.6 | 2.69 | 1.01 |
| minus1, round 0 | 41 | 861 | 12.7 | 3.04 | 2.28 |
| plus3, round 0 | 41 | 861 | 14.4 | 3.44 | 1.98 |
| tenrho, round 0 | 42 | 903 | 13.3 | 3.34 | 1.04 |
| cube, round 0 | 43 | 946 | 14.9 | 3.90 | 2.37 |
| h1, round 0 | 46 | 1,081 | 11.5 | 3.44 | 2.33 |
| mont, round 0 | 46 | 1,081 | 13.5 | 4.07 | 1.99 |
| fib, round 0 | 55 | 1,540 | 14.6 | 6.25 | 2.45 |
| rho, round 0 | 60 | 1,830 | 15.3 | 7.76 | 3.58 |
| affine, round 0 | 63 | 2,016 | 14.4 | 8.05 | 2.31 |
| tenrho, round 1 (6 routines more) | 45 | 1,035 | 25.7 | 7.38 | 3.72 |
| ten, round 1 (6 routines more) | 49 | 1,225 | 26.3 | 8.93 | 3.57 |
| plus3, round 1 (6 routines more) | 54 | 1,485 | 33.4 | 13.76 | 2.92 |
| cube, round 1 | 47 | 1,128 | 24.6 | 7.70 | – |
| mont, round 1 | 45 | 1,035 | 20.8 | 5.98 | – |
| fib, round 1 | 41 | 861 | 25.6 | 6.13 | – |
| PRACTICE.md: h1 with the doubled law book | 101 | 5,151 | 18.6 | 26.6 | 2.78 |
| PRACTICE.md: h1 at N ≤ 26 | 43 | 946 | 52.9 | 13.9 | 10.9 |

The lookahead hours grow with the square of L: 2.7 at L = 36, 8.1 at L = 63, 26.6 at L = 101. That is 1.3 to 3.5
times the search spent probing the candidates at L ≤ 63, and 10 times at L = 101. The seconds per lookahead almost
double once the first round's routines join the vocabulary. With a library carried from target to target L grows
further: the hardest-first sequence had 89 to 111 learnable problems a step (3,900 to 6,100 pairs), and the
pairs signal's own sequence was stopped after 5 hours on its second target.

**Cheaper rules, replayed.** Every lookahead's result is kept (`looks` in the round records), so any pair rule that
looks at a subset of the pairs can be replayed exactly, with no new search: which four problems it would pick, and
whether two of them are a pair that makes the target solvable in the lookahead (the condition under which the full
target search found it in every case here). The smallest k that keeps a solving pair, with the lookaheads and
search-hours it needs:

| target, round 0 | L | pairs | solving pairs | pairs among the k best singles | pairs among the k hardest | pairs with one member among the k best singles | pairs with one member among the k hardest | all pairs |
|---|---|---|---|---|---|---|---|---|
| h1 | 46 | 1,035 | 2 | k = 23 (299, 1.00 h) | k = 21 (210, 0.74 h) | k = 5 (261, 0.78 h) | **k = 2 (89, 0.29 h)** | 1,081 (3.44 h) |
| rho | 60 | 1,770 | 1 | k = 46 (1,095, 4.69 h) | k = 25 (300, 1.29 h) | k = 15 (840, 3.33 h) | **k = 2 (117, 0.46 h)** | 1,830 (7.76 h) |
| minus1 | 41 | 820 | 1 | k = 19 (212, 0.80 h) | k = 2 (1, 0.01 h) | k = 3 (158, 0.57 h) | **k = 2 (79, 0.37 h)** | 861 (3.04 h) |
| affine | 63 | 1,953 | 300 | k = 2 (64, 0.19 h) | k = 2 (1, 0.00 h) | k = 2 (186, 0.58 h) | **k = 2 (123, 0.40 h)** | 2,016 (8.05 h) |
| plus3, round 1 | 54 | 1,431 | 55 | k = 2 (55, 0.41 h) | k = 5 (10, 0.14 h) | k = 2 (159, 1.79 h) | **k = 2 (105, 1.39 h)** | 1,485 (13.76 h) |
| plus3, ten, tenrho, cube, mont, fib | 36–55 | 630–1,485 | 0 | – | – | – | – | 666–1,540 (2.7–6.3 h) |

Ranking problems one at a time fails, because a winning pair's members are individually useless:
- the square alone saves −4.3 bits on x² mod N;
- "x², plus one" alone saves −0.1 bits on x² + 1, so it ranks 46th of 60.

What the winning pairs share is that one member is among the two hardest problems the probes could learn: the link
(cost 6) for x² mod N, x² + 1 and x² − 1, the wrap step (cost 7) for 2x + 1.

**The cheaper rule, run.** `ct2a2_clean` (`breadth/cheap.py`: learner.run with one patched line) keeps the single
lookaheads and only the pairs with one member among the 2 hardest learnable problems, about 3L lookaheads instead of
L²/2. Run on all ten targets (round 0, sharing the probes):
- it invents the same four (h1, rho, minus1, affine), each with the same two pieces as the full pairs signal (only
  the two filler picks differ);
- it misses the same six;
- it chooses with 105 to 186 lookaheads (0.39 to 0.62 search-hours) instead of 666 to 2,016 (2.69 to 8.05): 6 to 14
  times less.

Caveat: PRACTICE.md's doubled law book has 13 problems tied at the top cost. Ties are broken by pool order, so there
the link may not be among the 2 hardest, and k would have to grow with the ties (not run).

**The trade-off.** Three ways to choose, by cost:
- *Hardest first*: no lookaheads at all. It misses formulas whose second piece is easy: x² + 1 in round 0, where all
  four picks were hard and the needed "x², plus one" costs 4.
- *All pairs*: catches any two pieces that work together, at L²/2 lookaheads, each slower as the library grows.
- *Pairs anchored on the 2 hardest*: catches a hard piece with any partner, which is every solving pair seen here,
  at about 3L lookaheads.

It would miss a formula that needs two easy pieces together and no hard one, or one whose hard piece ties with
many others at the top cost. Since the members of a solving pair are useless alone, any rule that scores problems
one at a time fails on these targets.

### 6. How hard each was tried

Budgets for every run are the learner's own (PRACTICE.md §9):
- probes: 30,000 states (120 s), then 1M states (600 s) for the unsolved;
- lookaheads: 30,000 states;
- practice and target searches: 3M states (2,400 s);
- loop bodies of cost ≤ 2, 4 picks a round, at most 5 rounds.

Every job's states and seconds are in the records (`report.py --effort`). A job answered from the shared cache counts
what it cost when first run, in every run that used it. Summed that way, the runs here count about 520
search-hours, plus about 60 container-hours for the unfinished x² + 1 step of the pairs-signal sequence.

| run | target | rounds finished | outcome | probe jobs (a / b) | lookaheads | practice searches | target searches | states | search-hours |
|---|---|---|---|---|---|---|---|---|---|
| lp_clean_affine | affine | 1 | general in round 0 | 216 / 130 | 0 | 0 | 1 | 573.4M | 2.32 |
| lp_clean_cube | cube | 5 | not found | 941 / 438 | 0 | 0 | 5 | 4,583.9M | 38.33 |
| lp_clean_fib | fib | 5 | not found | 843 / 343 | 0 | 0 | 5 | 3,206.9M | 31.24 |
| lp_clean_h1 | h1 | 1 | general in round 0 | 192 / 114 | 0 | 0 | 1 | 531.5M | 2.38 |
| lp_clean_minus1 | minus1 | 1 | general in round 0 | 192 / 127 | 0 | 0 | 1 | 558.4M | 2.29 |
| lp_clean_mont | mont | 5 | not found | 845 / 355 | 0 | 0 | 5 | 3,057.1M | 27.76 |
| lp_clean_plus3 | plus3 | 2 | general in round 1 | 368 / 216 | 0 | 0 | 2 | 1,495.6M | 6.29 |
| lp_clean_rho | rho | 2 | general in round 1 | 526 / 321 | 0 | 0 | 2 | 2,000.8M | 8.60 |
| lp_clean_ten | ten | 5 | not found | 840 / 401 | 0 | 0 | 5 | 4,354.4M | 46.71 |
| lp_clean_tenrho | tenrho | 4 | not found | 694 / 319 | 0 | 0 | 4 | 3,068.7M | 25.63 |
| ct2_clean_affine | affine | 1 | general in round 0 | 216 / 130 | 2016 | 0 | 1 | 704.5M | 10.37 |
| ct2_clean_cube | cube | 2 | not found | 381 / 240 | 2074 | 0 | 2 | 1,959.2M | 19.44 |
| ct2_clean_fib | fib | 2 | not found | 362 / 187 | 2401 | 0 | 2 | 1,317.4M | 17.36 |
| ct2_clean_h1 | h1 | 1 | general in round 0 | 192 / 114 | 1081 | 0 | 1 | 586.6M | 5.78 |
| ct2_clean_minus1 | minus1 | 1 | general in round 0 | 192 / 127 | 861 | 0 | 1 | 617.2M | 5.34 |
| ct2_clean_mont | mont | 2 | not found | 371 / 232 | 2116 | 0 | 2 | 1,739.8M | 15.37 |
| ct2_clean_plus3 | plus3 | 2 | general in round 1 | 321 / 180 | 2346 | 0 | 2 | 1,392.5M | 22.18 |
| ct2_clean_rho | rho | 1 | general in round 0 | 276 / 185 | 1830 | 0 | 1 | 987.3M | 11.35 |
| ct2_clean_ten | ten | 2 | not found | 247 / 123 | 1891 | 0 | 2 | 918.2M | 15.65 |
| ct2_clean_tenrho | tenrho | 2 | not found | 251 / 120 | 1938 | 0 | 2 | 855.8M | 14.62 |
| ct2a2_clean_affine | affine | 1 | general in round 0 | 216 / 130 | 186 | 0 | 1 | 577.3M | 2.90 |
| ct2a2_clean_cube | cube | 1 | not found | 190 / 123 | 126 | 0 | 1 | 597.2M | 2.90 |
| ct2a2_clean_fib | fib | 1 | not found | 219 / 126 | 162 | 0 | 1 | 623.5M | 3.12 |
| ct2a2_clean_h1 | h1 | 1 | general in round 0 | 192 / 114 | 135 | 0 | 1 | 521.9M | 2.75 |
| ct2a2_clean_minus1 | minus1 | 1 | general in round 0 | 192 / 127 | 120 | 0 | 1 | 566.4M | 2.78 |
| ct2a2_clean_mont | mont | 1 | not found | 206 / 134 | 135 | 0 | 1 | 676.8M | 2.49 |
| ct2a2_clean_plus3 | plus3 | 1 | not found | 181 / 121 | 120 | 0 | 1 | 539.6M | 2.47 |
| ct2a2_clean_rho | rho | 1 | general in round 0 | 276 / 185 | 177 | 0 | 1 | 864.7M | 4.22 |
| ct2a2_clean_ten | ten | 1 | not found | 119 / 64 | 105 | 0 | 1 | 277.6M | 1.45 |
| ct2a2_clean_tenrho | tenrho | 1 | not found | 127 / 64 | 123 | 0 | 1 | 274.5M | 1.51 |
| all_once_affine | affine | 1 | general in round 0 | 216 / 130 | 0 | 0 | 1 | 573.4M | 2.38 |
| all_once_cube | cube | 1 | not found | 190 / 123 | 0 | 0 | 1 | 613.8M | 2.76 |
| all_once_fib | fib | 1 | not found | 219 / 126 | 0 | 0 | 1 | 694.7M | 2.89 |
| all_once_h1 | h1 | 1 | general in round 0 | 192 / 114 | 0 | 0 | 1 | 514.8M | 2.41 |
| all_once_minus1 | minus1 | 1 | general in round 0 | 192 / 127 | 0 | 0 | 1 | 559.3M | 2.36 |
| all_once_mont | mont | 1 | not found | 206 / 134 | 0 | 0 | 1 | 715.5M | 2.41 |
| all_once_plus3 | plus3 | 1 | not found | 181 / 121 | 0 | 0 | 1 | 567.8M | 2.31 |
| all_once_rho | rho | 1 | general in round 0 | 276 / 185 | 0 | 0 | 1 | 858.7M | 3.71 |
| all_once_ten | ten | 1 | not found | 119 / 64 | 0 | 0 | 1 | 294.2M | 1.32 |
| all_once_tenrho | tenrho | 1 | not found | 127 / 64 | 0 | 0 | 1 | 317.6M | 1.46 |
| curated_clean_affine | affine | 1 | general in round 0 | 0 / 0 | 0 | 16 | 1 | 91.7M | 0.48 |
| curated_clean_cube | cube | 5 | not found | 0 / 0 | 0 | 18 | 5 | 446.5M | 3.65 |
| curated_clean_fib | fib | 5 | not found | 0 / 0 | 0 | 18 | 5 | 446.5M | 3.37 |
| curated_clean_minus1 | minus1 | 1 | general in round 0 | 0 / 0 | 0 | 16 | 1 | 153.7M | 0.74 |
| curated_clean_mont | mont | 5 | not found | 0 / 0 | 0 | 18 | 5 | 165.0M | 1.11 |
| curated_clean_plus3 | plus3 | 2 | general in round 1 | 0 / 0 | 0 | 18 | 2 | 226.9M | 1.62 |
| curated_clean_rho | rho | 1 | general in round 0 | 0 / 0 | 0 | 16 | 1 | 93.4M | 0.54 |
| curated_clean_ten | ten | 5 | not found | 0 / 0 | 0 | 18 | 5 | 446.5M | 3.56 |
| curated_clean_tenrho | tenrho | 5 | not found | 0 / 0 | 0 | 18 | 5 | 446.5M | 3.51 |
| seq_lp_clean_targets | h1 | 1 | general in round 0 | 192 / 114 | 0 | 0 | 1 | 531.5M | 2.38 |
| seq_lp_clean_targets | rho | 1 | general in round 0 | 299 / 24 | 0 | 0 | 1 | 101.3M | 1.85 |
| seq_lp_clean_targets | minus1 | 1 | general in round 0 | 222 / 14 | 0 | 0 | 1 | 203.7M | 6.04 |
| seq_lp_clean_targets | plus3 | 1 | general in round 0 | 266 / 20 | 0 | 0 | 1 | 391.0M | 9.73 |
| seq_lp_clean_targets | affine | 1 | general in round 0 | 162 / 35 | 0 | 0 | 1 | 138.4M | 6.88 |
| seq_lp_clean_targets | ten | 1 | not found | 238 / 128 | 0 | 0 | 1 | 475.6M | 25.15 |
| seq_ct2_clean_targets | h1 | 1 | general in round 0 | 192 / 114 | 1081 | 0 | 1 | 586.6M | 5.78 |
| branch: cube | cube | 1 | general in round 0 | 184 / 78 | 0 | 0 | 1 | 306.6M | 15.09 |
| branch: fib | fib | 1 | not found | 140 / 38 | 0 | 0 | 1 | 148.8M | 9.46 |
| branch: mont | mont | 1 | general in round 0 | 219 / 97 | 0 | 0 | 1 | 339.4M | 12.05 |
| branch: tenrho | tenrho | 1 | not found | 275 / 145 | 0 | 0 | 1 | 536.3M | 28.30 |

Stopped early, and why:
- **The pairs signal on x² + 3, 10·x², 10·x² + 1, x³, Montgomery's and Fibonacci**: after round 1 (2 rounds), by a
  per-run stop. Round 0 found no pair that solves any of them, and round 1 cost 6.0 to 13.8 lookahead-hours each.
  On x² + 3, round 1 solved it.
- **Practice-everything**: one round by design (`all_once`).
- **The cheaper pairs signal**: round 0 only by design (the comparison with the pairs signal's round 0).
- **The branches**: round 0 only. Their round 0 took 2 to 3.5 hours of wall-clock against a 34-token library.
- **The hardest-first chain**: stopped after 10·x²'s round 0, since the branches cover the hard targets. That round
  alone cost 25.2 search-hours with 41 routines.
- **The pairs-signal sequence**: stopped unfinished in its second target (x² + 1) after more than 5 hours on 12
  containers. The workspace was at 97 of its 100 containers, mostly mine.
- **Hardest first on 10·x² + 1**: stopped in its 5th round (round 4) after about an hour, to finish. Its best
  attempt over 4 rounds: 18, 18, 19, 21 of 28 pairs.

Infrastructure:
- Search containers were preempted a few times (Modal restarts the input).
- Runs were relaunched four times: to reorder the sequences, to give the hardest-first sequence its own
  containers, to consolidate drivers, and to add the per-run stop. The cache replays the finished searches.
- Cancelled or stopped drivers can write one last record. `report.load` reads a run's records from its last
  restart and drops a trailing round whose target search never ran: the 10·x² + 1 round cut off by the stop.
  Records written by the later launches carry their launch.
- The x² mod N gate was rechecked on the final records.

### 7. Files and how to rerun

Everything is in `cpu_first/bar3/breadth/`. It imports `bar3/`, `library/` and `library/inventing/` and changes
none of them.
- `targets.py`: the ten targets and `use(name)`, which points `library/inventing/target.py`'s `pairs` and `answers`
  at the calling thread's target.
- `carry.py`: learner.run with three patched lines (start from a carried state, return it), and `sequence`, the
  targets in turn, each solved target joining as a routine.
- `cheap.py`: learner.run with one patched line, the cheaper pairs signal (`anchor`: one member among the k best
  singles; `anchor_cost`: among the k hardest).
- `drive.py`: the runs, local or on Modal, through a tap that tags target jobs and keeps every lookahead (`looks` in
  each round record). `CONFIGS` is the learner's, plus `all_once` and `ct2a2_clean`.
- `modal_breadth.py`: Modal. Drivers run one run each, or many in one container, a thread each (`--multi`). Searches
  fan out to CPU containers and identical jobs share a `modal.Dict` with in-flight claims; stale claims are retaken.
  A per-run stop sets the rounds a run may finish (`--stop NAME=R`). A sequence saves its state after each target,
  and `--branch` runs targets from such a state. Limits: every search has the learner's state and time budget,
  search containers a 3-hour timeout and a 32 GiB hard memory limit, drivers 24 hours.
- `report.py`: the tables here. `--matrix`, `--fresh`, `--seq`, `--pairs` (the exact replay of cheaper pair rules),
  `--effort`, `--same` (the gate against PRACTICE.md), and a round-by-round `--detail`.
- `records/<run>.jsonl`: one line per round, PRACTICE.md's fields plus `looks`, `target_name` (and `carry`,
  `carried_lib` in sequences). `<seq>.state.<target>.json`: a sequence's carried state after each target.
  `calls_*.json`: the Modal call ids. `local_tiny.jsonl` and `*_smoke*.jsonl`: the plumbing checks (k = 5, tiny
  budgets).

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.breadth.drive --targets rho,fib
```

```bash
cd cpu_first && BREADTH_MAX=40 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/breadth/modal_breadth.py --multi --groups 2 --fresh lp_clean,ct2_clean --targets rho,minus1,affine --tag rerun
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/breadth/modal_breadth.py --multi --seq lp_clean --order h1,rho,minus1,plus3,affine --tag rerun
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/breadth/modal_breadth.py --fetch
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.breadth.report --matrix bar3/breadth/records/*_b1.jsonl
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.breadth.report --pairs bar3/breadth/records/ct2_clean_*_b1.jsonl
```

Search jobs are cached by content in `bar3-breadth-cache`, so a rerun of an unchanged config replays the stored
records in minutes; bump `VERSION` in `modal_breadth.py` to search again from scratch.
