# sp2's law book and choosing rule, one part at a time

*Written 2026-10-24 (PT). Code, plans and records are in [`integration/sp_ablation/`](.). The model is self-posed
practice's final file, [`submission_stage3_sp2.py`](../self_practice/submission_stage3_sp2.py) (sp2, md5
`afc70fe91069a85c416eba18b48ee30e`, [SELF_PRACTICE.md](../SELF_PRACTICE.md)), which this study never edits.*

## In plain words

<!-- PLAIN -->
**The question.** sp2 learns x^(2^T) mod N in one training run without a curriculum a person chose. It poses its own
practice problems from a book of 11 laws (compare, equal, min, max, remainder, quotient, add, subtract, multiply,
multiply-add, add three) applied to 8 patterns of the data's numbers (for example "x + d N": x plus a digit's worth of
N), in every placement of the operands in the registers, and a choosing rule (`lpca`) picks what to practise. A judge
will say that both are designed: the book lists the operations the answer needs, and the rule was picked by watching
whether the answer came. So each law and each pattern was taken out of the book in turn, the rule was swapped for
every rule bar 3 tried and for two that take `lpca` apart, and every arm was trained again as the contest does it: one
H100, 3,600 s, the untouched official manifest, seed 74, then seed 1. The unchanged file, run beside them, replays
sp2's official record exactly.

**What the result needs** (on both seeds, 74 and 1).
- **The multiply-add law, a + b c, and its pattern (another row's x, x, a digit).** Without either there is no row
  that adds x times a digit into a running sum, and the multiply is a loop over that row, so there is no multiply. A
  plain row ("x times a digit", which clears the sum first) is still posed and practised, and does not stand in.
- **What every other problem is built or checked with: compare, remainder, add, and the pair pattern.** Every number
  but the pair's is built with the learner's own addition, and the addition's practice problem is checked with its
  own comparison and remainder. Without "compare" the learner still had an identical program (subtraction's borrow),
  but the check asks for "compare" by name, so nothing past the pair was ever posed.
- **The round's own two routines**: the multiply (on x and another row's x) and the reduction (on x + o N). The round
  is a few routine calls with no loops, so without either there is no round at any budget.
- **In the rule, rating a law on a pattern by its cheapest placement.** bar 3's hardest-first, which rates single
  candidates, practises awkward placements of easy laws first and never gets the reduction; every other rule bar 3
  tried fails too, except practising everything (below).
- **Two more parts of lpca make the result reliable rather than possible.** Each failed on seed 74 and certified on
  seed 1. Hardest first among the concepts: in a random order the multiply-add row, which can be found at one level
  only, is kept or lost by the draw. Every placement tied for cleanest: with one placement per concept, whether the
  reduction lands in registers the round can use is chance (on seed 1 the learner found a long division sp2 never
  had, one that shifts the divisor instead of the dividend).

**What made it slower on seed 74 and failed on seed 1.** Nothing in the book or the rule costs only time on both
seeds.
- **The x + d N pattern** (the chain's practice problem): without it the reduction must write the chain inline, a
  loop of subtractions inside the loop over x's digits, at the deep search's last level. On seed 74 it was found a
  level later (+441 s); on seed 1 never, although the program is right there too.
- **Choosing at all**: practising every learnable candidate (bar 3's `all`) certifies on seed 74 at the edge of the
  budget (the round by 3,211 s, frozen by 3,249 s of the 3,600). On seed 1 it runs out of time a level short.

**What it doesn't need**, each on its own: six of the 11 laws (equal, min, max, quotient, subtract, add three), three
of the 8 patterns (x + N, 10 x + d, x with a digit), and the choice of the cleanest placements over the cheapest. Seed 1
agrees on every one of them.

**A smaller book.** All of those removed at once, and x + d N too (picked on seed 74, where it only cost time): 5
laws on 4 patterns, 6 concepts. It certifies on the official manifest sooner than sp2 (the round by 571 s against
977 s) and on all 8 seeds (443-579 s on a B200). That includes seed 1: there it writes the chain inline, which sp2's
book without x + d N did not. Its climb is nothing but the round's pieces, one a level: compare,
subtract-if-it-fits, add, the multiply-add row, the multiply, the reduction.

**A generated book.** The book written as a grammar instead of a list: numbers r + q b from the division law (x and N,
a digit and the base; q one, a digit, another row's x, or x), the machine's binary arithmetic and every composition of
+ and ×, every operation on every pattern, one set of target rules. It holds sp2's book (the same targets on all of
sp2's problems) and 24 concepts more. It certifies on the official manifest (the round by 2,198 s, frozen by 2,228 s)
and on all 8 seeds (1,675-2,084 s on a B200), by three routes sp2's book does not offer. For example: the square as
"0 plus x times x" through a generated multiply-add of two numbers, or as its own generated concept, "x times x".

**What this says about bar 2.** The particular list of laws is not what the result rests on: a book generated from a
few rules, which picks no law, pattern or composition by hand, also yields the round. What it needs are ingredients
any such grammar must have: a composition of + and × that accumulates (the multiply-add row), the division law's form
(to pose reductions), and target rules from facts, identities and the learner's own routines. Still designed: that
grammar (its atoms, its forms, the patterns of the comparison pair), the target rules, the choosing rule, and the rest
of sp2 (SELF_PRACTICE.md §6).

**How firm this is.** Seed 74 ran for all 30 arms and both books ran on all 8 seeds. Seed 1 ran for 13 arms before
Modal stopped the workspace at its spend limit; 17 seed-1 arms and a rerun at three times the budget are not run yet
(about $82 to finish, §6). Most failures were cut by the training deadline, not by running out of things to try; for
each, §4 says why more time would not help: by construction, two full levels that found nothing, a piece that can be
found at one level only, or longer runs of the same rule elsewhere. The exception is one placement per concept, whose
rerun at three times the budget had no round when it was stopped at 6,229 s.
<!-- /PLAIN -->

## 1. What was removed, and how

sp2 is stage 3 (s3i) with its designed curriculum taken out. Instead of nine practice problems a person chose, the
learner poses its own: every law of a small book applied to numbers built from the data's own N and x, in every
placement of the operands in the registers, and a general rule picks which to practise. Two things in that are still
designed (SELF_PRACTICE.md §6), and this study removes them piece by piece:
- **the law book**: 11 laws (compare, equal, min, max, remainder, quotient, add, subtract, multiply, multiply-add,
  add three) on 8 patterns of the data's numbers: the pair (x and N, both orders, with itself, with zero), x + N,
  x + d N (d a digit), x + o N (o another row's x), 10 x + d, (x, d), (x, o), (o, x, d);
- **the choosing rule** `lpca`: a concept (a law on a pattern) is rated by its cheapest placement, the hardest K = 4
  concepts are practised each level, each once in every placement tied for cleanest (fewest registers touched).
  It was picked after bar 3's rules failed, by watching whether the round came.

Each arm removes one part and keeps everything else as it is.
- **The book's arms** pass SelfClimb its own `laws=` and `patterns=` arguments (sp2 passes neither: the whole book), as
  tuples in the book's order, so the pose order and its ties are sp2's: one law out of the book (on every pattern), or
  one pattern out (every law on it).
- **The rules sp2 already has** (`lp`, `ct2a2`, `all`, `lpu`, `lpc`) run sp2 itself with `SP_RULE` rewritten by the
  shared runner's `--set`.
- **The other rules** are variants with a few lines in `pick` (and, for `ct1`, the lookahead's job list):
  - `random_ok`, `ct1` and `nov` are bar 3's (bar3/PRACTICE.md §3): K learnable candidates at random; the K whose
    routine alone shortens the round's search most; the K whose program's answers on the target's inputs differ most
    from every token's.
  - `rca` and `lpcc` take `lpca` apart. `rca` takes its concepts in a random order instead of hardest first.
    `lpcc` practises each concept in its cheapest placements instead of its cleanest.

  The random rules draw from a generator seeded by the run's seed (the evaluator's `torch.manual_seed`) and the level.
  bar 3's `random` (no probes; each random pick given a full search) has no counterpart here: sp2's only searches are
  its probes, so a pick the probes did not solve has no program to practise, and `random_ok` is the nearest arm.

[`make_variants.py`](make_variants.py) builds every variant: one comment line at the top, then a few text edits of
sp2, each asserted to occur exactly once. `make_variants.py --diff ARM` prints an arm's edits; the md5s are in
[`variants/MD5SUMS.json`](variants/MD5SUMS.json). [`check_variants.py`](check_variants.py) checked every arm locally
before any run (30 arms, about 30 s, under 1 GB):
- each book arm, given the routines the book's constructions need (sp2's official run's own adder, link, comparison
  and chain), poses exactly the control's pool minus the dropped law's or pattern's candidates;
- each rule arm classifies and chooses on one shared toy level (a real probe of the control's at toy budgets), and its
  picks are what the rule says (random_ok K distinct learnable; lpcc every placement tied for cheapest; ct1 a
  lookahead per learnable; nov the novelty scores' top K).

<!-- ARMS -->
| arm | the designed part | how it is removed | md5 of what ran |
|---|---|---|---|
| control | none: sp2 as certified | the file, untouched | `afc70fe9` |
| rule_lp | the rule | `--set SP_RULE="lp"`: bar 3's hardest learnable first (candidates, not concepts) | `3283c0ee` |
| rule_ct2a2 | the rule | `--set SP_RULE="ct2a2"`: bar 3's cheaper pairs signal | `bd8d0f90` |
| rule_all | the rule | `--set SP_RULE="all"`: every learnable candidate practised (bar 3's reference) | `35eeb33c` |
| rule_lpu | the rule | `--set SP_RULE="lpu"`: unlockers first, then hardest first (SELF_PRACTICE.md §3.4) | `58990dec` |
| rule_lpc | lpca's ties | `--set SP_RULE="lpc"`: each concept in its one cleanest placement | `e103b0eb` |
| no_law_lt | the book: a law | variant: compare (lt) out of the book, on every pattern | `182a689d` |
| no_law_eq | the book: a law | variant: equal (eq) out of the book, on every pattern | `dd3a6366` |
| no_law_min | the book: a law | variant: min out of the book, on every pattern | `490167c3` |
| no_law_max | the book: a law | variant: max out of the book, on every pattern | `761035e0` |
| no_law_red | the book: a law | variant: remainder (red) out of the book, on every pattern | `dcb4fa96` |
| no_law_div | the book: a law | variant: quotient (div) out of the book, on every pattern | `eb247352` |
| no_law_add | the book: a law | variant: add out of the book, on every pattern | `454f47b3` |
| no_law_sub | the book: a law | variant: subtract (sub) out of the book, on every pattern | `d83686c6` |
| no_law_mul | the book: a law | variant: multiply (mul) out of the book, on every pattern | `3736584b` |
| no_law_accmul | the book: a law | variant: multiply-add a + b c (acc.mul) out of the book, on every pattern | `2e09afe3` |
| no_law_accadd | the book: a law | variant: add three a + b + c (acc.add) out of the book, on every pattern | `71f7fd21` |
| no_pat_pair | the book: a pattern | variant: the pair (x and N) out of the book, every law on it | `c3561295` |
| no_pat_xn | the book: a pattern | variant: x + N out of the book, every law on it | `f8b2cab0` |
| no_pat_xdn | the book: a pattern | variant: x + d N out of the book, every law on it | `c4a0705a` |
| no_pat_xon | the book: a pattern | variant: x + o N out of the book, every law on it | `b723f191` |
| no_pat_bxd | the book: a pattern | variant: 10 x + d out of the book, every law on it | `8e47c3af` |
| no_pat_xd | the book: a pattern | variant: (x, d) out of the book, every law on it | `0c43a0de` |
| no_pat_xo | the book: a pattern | variant: (x, o) out of the book, every law on it | `17aafbfe` |
| no_pat_oxd | the book: a pattern | variant: (o, x, d) out of the book, every law on it | `0c86793b` |
| rule_random_ok | the rule | variant: bar 3's random_ok, K learnable candidates at random | `f407fa64` |
| rule_rca | lpca's order | variant: lpca's concepts and placements, in a random order | `e19291f3` |
| rule_lpcc | lpca's cleanest placements | variant: each concept in every placement tied for cheapest | `328124e3` |
| rule_ct1 | the rule | variant: bar 3's ct1, the K best single lookaheads | `7ba907ac` |
| rule_nov | the rule | variant: bar 3's novelty, on the target's inputs | `38db4eb1` |

The two books (§5) are built the same way by [`make_books.py`](make_books.py): `book_min` (`e4732f09`) and `book_gen`
(`8bbac4a1`).
<!-- /ARMS -->

## 2. Results

<!-- RESULTS -->
Every run is the untouched official manifest on one H100 (`benchmark/manifests/h100_hard_h1.json`; seed 1:
`runtime.seeds` = [1], the only change), so a certifying arm's seed-74 run is an official run. "Certified" means T =
64 on both ladders, every rung 768/768 and every split exact; every certified run here is. The round's time is the
evaluator's elapsed seconds at the first logged step after it, host-corrected in brackets where at least 10 search
units equal the control's (§4); "cut by the training deadline in L*n*" means the climb was still going when the
training deadline stopped it, "ended" that it ended by its own rule. Records: `records/<arm>_h100_seed<seed>.json`;
the table comes from [`analyze.py`](analyze.py) `--both`.

| arm | what it removes | seed 74 | seed 1 | verdict |
|---|---|---|---|---|
| control | none (sp2 as certified) | yes: 688 (L4), 977 s | yes: 630 (L4), 834 s | replays sp2's official record exactly |
| no_law_eq | the law eq | yes: 688 (L4), 785 s | yes: 630 (L4), 737 (812) s | not needed |
| no_law_min | the law min | yes: 688 (L4), 1,119 s | yes: 630 (L4), 849 (863) s | not needed |
| no_law_max | the law max | yes: 688 (L4), 833 (971) s | yes: 637 (L5), 978 (896) s | not needed |
| no_law_div | the law div | yes: 688 (L4), 950 (938) s | yes: 631 (L4), 866 (855) s | not needed |
| no_law_sub | the law sub | yes: 688 (L4), 906 (1,016) s | yes: 630 (L4), 749 (811) s | not needed |
| no_law_accadd | the law a + b + c | yes: 686 (L4), 941 (975) s | yes: 628 (L4), 847 (831) s | not needed |
| no_pat_xn | the pattern x + N | yes: 685 (L4), 974 (977) s | yes: 627 (L4), 861 (817) s | not needed |
| no_pat_bxd | the pattern 10 x + d | yes: 687 (L4), 830 (942) s | yes: 629 (L4), 877 (827) s | not needed |
| no_pat_xd | the pattern (x, d) | yes: 686 (L4), 850 (912) s | yes: 628 (L4), 857 (777) s | not needed |
| rule_lpcc | lpca's cleanest placements (cheapest instead) | yes: 688 (L4), 1,063 (1,008) s | yes: 630 (L4), 926 (841) s | not needed |
| no_pat_xdn | the pattern x + d N | yes: 694 (L5), 1,302 (1,418) s | no: cut by the training deadline in L9 (of L0-L10) | speed only on seed 74 (+441 s); needed on seed 1 |
| rule_all | choosing at all (every learnable practised) | yes: 687 (L4), 3,211 s | no: cut by the training deadline in L4 (of L0-L14) | speed only on seed 74 (+2,234 s, at the budget's edge); needed on seed 1 (out of time) |
| rule_rca | lpca's hardest first (random order) | no: cut by the training deadline in L7 (of L0-L8) | yes: 629 (L4), 858 (962) s | needed on seed 74; not needed on seed 1 |
| rule_lpc | lpca's every tied placement (one each) | no: cut by the training deadline in L11 (of L0-L12) | yes: 640 (L5), 1,078 s | needed on seed 74; speed only on seed 1 (+244 s) |
| no_law_accmul | the law a + b c | no: cut by the training deadline in L8 (of L0-L10) | no: cut by the training deadline in L6 (of L0-L8) | needed |
| no_pat_oxd | the pattern (o, x, d) | no: cut by the training deadline in L8 (of L0-L10) | no: cut by the training deadline in L6 (of L0-L8) | needed |
| no_law_lt | the law lt | no: ended in L16: the level cap (step 737, 1,530 s) | no: ended in L16: the level cap (step 679, 890 s) | needed |
| no_law_red | the law red | no: ended in L16: the level cap (step 735, 1,450 s) | no: ended in L16: the level cap (step 676, 1,719 s) | needed |
| no_law_add | the law add | no: ended in L16: the level cap (step 735, 1,669 s) | no: ended in L16: the level cap (step 677, 1,198 s) | needed |
| no_pat_pair | the pattern pair | no: ended in L1: nothing learnable (step 647, 80 s) | no: ended in L1: nothing learnable (step 589, 75 s) | needed |
| no_law_mul | the law mul | no: cut by the training deadline in L9 (of L0-L10) | no: cut by the training deadline in L7 (of L0-L8) | needed |
| no_pat_xo | the pattern (x, o) | no: cut by the training deadline in L9 (of L0-L10) | no: cut by the training deadline in L7 (of L0-L8) | needed |
| no_pat_xon | the pattern x + o N | no: cut by the training deadline in L10 (of L0-L12) | no: cut by the training deadline in L8 (of L0-L10) | needed |
| rule_lp | lpca (bar 3's hardest first) | no: cut by the training deadline in L10 (of L0-L16) | no: cut by the budget's end in L9 (training 3,673 s) | needed |
| rule_ct2a2 | lpca (bar 3's cheaper pairs) | no: cut by the training deadline in L12 (of L0-L16) | no: cut by the training deadline in L8 (of L0-L16) | needed |
| rule_lpu | lpca (unlock first) | no: cut by the training deadline in L9 (of L0-L16) | no: cut by the training deadline in L8 (of L0-L16) | needed |
| rule_random_ok | lpca (bar 3's random_ok) | no: cut by the training deadline in L12 (of L0-L16) | no: cut by the training deadline in L8 (of L0-L16) | needed |
| rule_ct1 | lpca (bar 3's ct1) | no: cut by the training deadline in L8 (of L0-L16) | no: cut by the training deadline in L9 (of L0-L16) | needed |
| rule_nov | lpca (bar 3's novelty) | no: cut by the training deadline in L8 (of L0-L16) | no: cut by the training deadline in L6 (of L0-L16) | needed |
<!-- /RESULTS -->

## 3. Arm by arm

The control's climb, for reference (sp2's official run's, line for line; the concepts each level practised, each in
its tied-cleanest placements):
- **level 0:** lt on the pair (the borrow, `CLC P[SBB.Y INC]`), sub on the pair;
- **level 1:** min and red on the pair (the link, `STA.S T.lt.pair.AY SEL.S`);
- **level 2:** max and eq on the pair, add on the pair (`CLC P[ADC.Y INC]`);
- **level 3:** acc.mul on (o, x, d) (stage 3's multiply-accumulate row exactly, `LDH.Y CLK P[MUL.X INC]`), red and
  sub on x + N, red on x + d N (the chain, `B[T.red.pair.AY]`);
- **level 4:** mul on (x, o) (the multiply, `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]`), red on x + o N (the reduction,
  `STA.Y P[BRG.X T.red.x+dn.AY]`: long division over the chain), red on 10 x + d (the step), mul on (x, d) (the row);
- **the round** at step 688: `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA`, which calls the multiply, the
  reduction, the multiply-accumulate row, the chain, the link and the comparison.

<!-- ARM_BY_ARM -->
**Not needed.** Each of these certifies with the round at step 685-688 (level 4), and the same round program,
`LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA`, with the same routines under it. The levels' freed slots go to
what the control would have practised a level later, or to other laws:
- **eq, max** (no_law_eq, no_law_max): level 2 practises only add (no_law_eq: max was not among its learnable
  candidates there), or eq and add (no_law_max). The round at step 688 in both.
- **div** (no_law_div), **a + b + c** (no_law_accadd): the climb practises the same concepts level for level as the
  control; the quotients and a + b + c were never practised there either. The round at step 688 and 686.
- **sub** (no_law_sub): level 0 practises only lt; level 3 practises lt on x + d N instead of sub on x + N.
- **min** (no_law_min): the round at the same step (688), 142 s later. Without min, max has no short program (the
  control builds it from min at level 2), so max stays open, wins level 3's deep probes and takes 165 s of them
  without a program; in the control those probes went to x + d N and took 12 s. A real cost, within host noise.
- **x + N** (no_pat_xn): level 3 practises lt on x + d N and a + b + c instead of red and sub on x + N.
- **10 x + d** (no_pat_bxd, the long-division step's pattern) and **(x, d)** (no_pat_xd, the multiply's plain row):
  level 4 practises a + b + c instead. sp2's round never calls the step or the plain row (its reduction runs the
  chain directly, and its multiply loops over the accumulating row).
- **The cheapest placements instead of the cleanest** (rule_lpcc): the same concepts at the same levels; on this seed
  the cleanest placements of the round's pieces are also their cheapest.

**Speed only on seed 74, failed on seed 1.**
- **x + d N** (no_pat_xdn, the chain's pattern). On seed 74 the round comes at step 694 (level 5), 441 s later
  (host-corrected). With no chain to practise, the reduction comes as `STA.Y P[BRG.X B[T.red.pair.AY]]`: long
  division with the chain written inline, a loop of links inside the loop over X's digits (cost 5). The deep probe of
  level 5 finds it (193 s). On seed 1 it never comes. The climb has every other piece by level 4 (the link, the row,
  the multiply). Levels 5 and 7 pose only the two concepts not yet practised, red and the quotient on x + o N (24
  candidates), and deep-probe all 12 placements to cost 5 in full (337 s and 480 s): nothing is learnable. Level 8
  tries again, and the training deadline cuts level 9's deep probes. The program itself is not at fault: it is right
  with seed 1's own routines at every size tested ([`reduce_check.py`](reduce_check.py)). It was not among the
  programs those searches could build. Their vocabulary is the routines the probes' closest attempts used (20-23 at
  seed 74's level 5, 15-20 on seed 1). The search also merges items that agree on its random test machines, so the
  loops it holds depend on that vocabulary. Why this one was missing was not looked into further. With the chain
  practised as a routine of its own, the reduction costs 4 and comes at level 4 on both seeds; written inline, it
  sits at the deep search's last level and comes on one seed of two. The smaller book (§5.1) also leaves x + d N
  out, and there the inline chain came on all 8 seeds. So what one removal costs depends on what else is in the book.
- **bar 3's all** (every learnable candidate practised, no choice at all). On seed 74 it certifies: the round at step
  687 (level 4) by 3,211 s, frozen by 3,249 s, with 3,412 s of training. That is 269 s before the climb's deadline
  (3,480 s, 120 s before the budget's end). It practises 112 candidates in 5 levels (the control 47), so level 4 alone
  takes 2,450 s, one of its probes stopped by the 900 s search limit. Its round is 5 items,
  `LDA.X STA.Y T.mul.x,o.XY STA.X T.red.x+on.XN`: having practised every placement, it has the reduction with the
  modulus in N, where sp2 has it in A and loads it. On seed 1 it runs out of time. Levels 0-2 leave a larger library
  (level 3's round search has 118 one-token items, against 105 on seed 74), and level 3's probes take 1,337 s
  (seed 74: 471 s). The training deadline then cuts the climb in level 4, after one of its probes had used the whole
  900 s search limit. That is the level whose round came at 3,211 s on seed 74. So a rule that chooses is needed for
  time, not for discovery, and on one seed of two practising everything does not fit the hour. The rules that fail
  on both seeds are the ones that choose badly.

**Needed: the multiply-accumulate row.**
- **a + b c** (no_law_accmul) and **its pattern (o, x, d)** (no_pat_oxd): no multiply-accumulate row, so no multiply.
  Both climbs practise the comparison, the link, the addition, the chain and the reduction (level 4) as the control
  does. Then levels 5 and 7, which pose only concepts not yet practised (48 candidates: the multiply on (x, o) and
  (x, d) in every placement, and the quotients), are probed and deep-probed in full (623-801 s each) and nothing is
  learnable; the levels between practise other placements of what they have (eq, the reduction). The multiply loops
  over a row that adds into a running sum, and only a + b c poses that row; a row that clears A first (the plain row,
  mul on (x, d)) cannot accumulate within a loop body of 3. s3i's ablation found the same (mac_n). The training
  deadline cut both climbs in level 8, practising placements. On seed 1 the same, with one level of new concepts
  instead of two. The reduction comes at level 4 in another placement, `CLA P[BRG.X T.add.pair.AN T.red.x+dn.AN]`
  with the modulus in N, which is right at every size tested ([`reduce_check.py`](reduce_check.py)). Level 5 (the
  same 48 candidates) is probed and deep-probed in full (912-963 s) and nothing is learnable. The deadline cuts both
  climbs in level 6, before level 7 can pose them again.

**Needed: the parts every other problem is built or checked with.** Without them nothing past the pair pattern is posed.
- **lt** (no_law_lt): the learner does not lack a comparison: it practises sub on the pair, whose program
  `CLC P[SBB.X INC]` is the same borrow ripple, and builds a true link on it (`STA.S T.sub.pair.AX SEL.S`). But the
  addition's check calls the practised lt by its law's name, so the addition is never posed, and with it nothing that
  is built by addition (every pattern but the pair). The climb runs dry by its own rule: levels 3 and 5 pose nothing,
  the others other placements, to the level cap (16 levels by 1,530 s). Needed by the check's construction, not by
  what the learner can compute.
- **red** (no_law_red): the same. The addition's check also calls a practised red (red(a + N) = a), so no addition is
  posed, and nothing past the pair; levels 0-2 practise lt, sub, min, max and eq, level 3 poses nothing, and the
  climb runs dry to the level cap (by 1,450 s).
- **add** (no_law_add): every number but the pair's own is built by the learner's addition (x + N, x + d N, x + o N),
  and so is every multiply's target, so nothing but the pair is ever posed: the pair's six other laws in their
  placements, levels 3, 5 and 7 posing nothing, to the level cap (by 1,669 s).
- **the pair pattern** (no_pat_pair): lt, red and add are posed only there, and every other pattern is built with
  the addition, so level 0 poses nothing, and the climb ends by its own rule after 80 s (nothing learnable).
- **Seed 1 the same**: the lt, red and add arms run dry to the level cap (by 890-1,719 s), and the pair arm ends by
  its own rule after 75 s.

**Needed: the round's own two routines.** Without them nothing can be the round.
- **mul** (no_law_mul): the climb practises every other piece of the round by level 4 (the row, the chain, the
  reduction), and levels 5 and 7, which pose only the quotients (24 candidates, the concepts not yet practised), find
  nothing; the deadline cut it in level 9. With no multiply routine there is no round at any budget: the round's
  search composes routine calls and straight-line moves, no loops, and no other law's routine multiplies two
  numbers.
- **x + o N** (no_pat_xon, the reduction's own practice problem): every other piece comes as in the control (the
  multiply, the step and the row at level 4), and levels 5, 7 and 9, which pose only the 12 quotients on x + d N,
  find nothing; the deadline cut it in level 11. No practised routine reduces a number of 16 places (the chain and
  red on x + N handle a few multiples of N; the step one digit), and the round's search has no loops, so there is
  no round at any budget.
- **(x, o)** (no_pat_xo, the multiply's own practice problem): the same course as no_law_mul. The plain row (mul on
  (x, d)) is still posed and practised, but nothing multiplies two numbers, so there is no round at any budget.
- **Seed 1 the same.** mul and (x, o): the reduction at level 4 (in N, as under a + b c above), the 24 quotients of
  levels 5 and 7 unsolved, the deadline in level 7. x + o N: the multiply at level 4, the 12 quotients of levels 5
  and 7 unsolved, the deadline in level 8.

**Needed, in the rule.** lpca has four parts: concepts rated by their cheapest placement (against bar 3's
hardest-first on candidates, `lp`), hardest first among them (against a random order, `rca`), every placement tied
for cleanest (against one, `lpc`), and cleanest rather than cheapest (`lpcc`: not needed, above). The first is needed
on both seeds. The second and third failed on seed 74 and certified on seed 1, so they are needed for a reliable
result rather than a possible one. And choosing well matters: every other rule bar 3 tried fails on both seeds. The
exception is practising everything, which certifies on seed 74 at the budget's edge and runs out of time on seed 1.
- **bar 3's hardest learnable first** (rule_lp): the costliest candidates are costly placements of easy laws (the
  comparison and subtraction at cost 5 with their operands in X, Y or N, then min, max and the link), so levels 0-3
  go to the pair pattern, the addition comes at level 4, the multiply-accumulate row at 5, and the multiply and a
  chain at 7, but no reduction; 10 levels of real work (3,480 s), then the deadline. The same course as
  SELF_PRACTICE.md's tiny core, where at 16 probes a setting it had no reduction by level 10 either (3,976 s on a
  B200). Seed 1 the same: the addition at level 3, the row at 4, the multiply at 6, and chains at 7 that read or
  write X, where sp2's long division keeps the dividend. No reduction by the end (level 9, whose last search ran past
  the budget's end: §4).
- **Hardest first among the concepts** (rule_rca: lpca's concepts in a random order): the multiply-accumulate row is
  learnable only at level 3, where its roots get their one search over the primitives alone (cost 5 at 1M states).
  Taken in a random order, level 3 practises the chain, red on x + N, lt on x + d N and a + b + c instead. From level
  4 the row's roots are searched with the whole library, which does not reach cost 5, and every deep probe of levels
  4-7 goes to the closest open concept, the quotient on x + d N (closeness 0.23-0.36, never solved), never to the row
  or the multiply. So on seed 74 the row is lost for good, and the multiply with it; more time repeats the same
  levels. On seed 1 the same rule certifies (the round at step 629, level 4, about as fast as the control): its random
  draw at level 3 happened to include the row (with sub on x + N, the chain and a + b + c). Level 3 has 6 learnable
  concepts on both seeds and practises 4, so a random order keeps the row two times in three; hardest first keeps it
  every time, because the row is the costliest concept there. Needed for the result to be reliable, not for it to be
  possible.
- **Every placement tied for cleanest** (rule_lpc: each concept once, in its one cleanest placement): on seed 74 the
  comparison, the link and the chain are practised only against X (levels 0, 1, 4; the link and the comparison get
  placements against N at level 7, the chain never). sp2's long division brings X's
  digits into A one at a time (BRG.X), so its chain must take the modulus from Y or N, and with only the X chain that
  reduction is out of reach (s3i's no_twins result, in the self-posed setting). The multiply comes at level 8, and the
  deadline cut the climb in level 11 with no reduction. **On seed 1 the same rule certifies** (the round at step 640,
  level 5, 244 s after the control), with a reduction sp2 never found: `T.red.x+on.YX` =
  `T.mul.x,o.XY LDA.Y W[SHR.X T.red.x+dn.AX]`. It calls the multiply only for its side effect, which shifts X (the
  modulus) up 8 places, loads the dividend from Y, and then 8 times shifts the modulus down one place and runs the
  chain against X: long division by moving the divisor instead of bringing in the dividend's digits, which works with
  the X chain. The round is `LDA.X STA.Y T.mul.x,o.XY STA.Y LDA.N STA.X T.red.x+on.YX`. The rerun at 3 times the
  budget on seed 74 (§4) found the same reduction at level 11 (4,275 s), but with the dividend in N, which no
  instruction can write, so no round can use it; it had no round by level 15 (6,229 s) when Modal's spend limit
  stopped it. So one placement per concept makes the round a matter of which placement comes up, where every tied
  placement gave it on both seeds.
- **bar 3's cheaper pairs signal** (rule_ct2a2): 678 lookaheads, the round searched with each learnable candidate and
  with each pair that includes one of the two hardest, and none reaches the round (at full size it is reached or not,
  so no lookahead tells candidates apart until both the multiply and the reduction exist). With every gain zero it
  takes the two hardest and their first partners: levels 2-9 go to placements of the pair's laws, the addition comes
  at level 10, the row at 11, the multiply at 12; no chain, no reduction. On seed 1 it gets further, and still has
  no reduction. The addition comes at level 3, the row at 4, the multiply at 5. Chains come at 7, but they load the
  number from X or Y (`LDA.X B[T.max.pair.AY T.lt.pair.XY SEL.S]`). The deadline then cuts the climb in level 8.
- **bar 3's random_ok** (K learnable candidates at random): 12 levels of random placements of the pair's laws. Its
  first links are the alias "x if x < N, else 0" (`T.lt.pair.YX LDA.Y STA.S CLA SEL.S`), which cannot confirm an
  adder; a true link comes at level 6, but the addition is never among its random picks, so nothing past the pair
  pattern is ever posed. On seed 1 its draws include the addition (levels 3-4) and chains (levels 4-7). They never
  include the row, so there is no multiply and no reduction when the deadline cuts it in level 8.
- **SELF_PRACTICE.md's unlock first** (rule_lpu): the costliest link is the alias again (level 1), the addition comes
  at level 4, a chain at level 5 that uses X as scratch (`B[T.min.pair.AY T.lt.pair.XY SEL.S]`), the multiply at
  level 7, and no reduction by the deadline (level 9): SELF_PRACTICE.md §3.4's result again. Seed 1 the same. The
  addition comes at level 2, the row and a chain against X (`B[T.max.pair.AX T.lt.pair.YX SEL.S]`) at 3, and the
  multiply at 5. At 8 come chains that read the number from X. There is no reduction by the deadline (level 8).
- **bar 3's ct1** (the K candidates whose routine alone shortens the round's search most): 281 lookaheads, none of
  which reaches the round, so every gain is zero and the ties go by pool order: 8 levels of placements of the pair's
  laws, the addition at level 4, no row, no multiply, no chain, no reduction. Seed 1 the same: the addition at level
  4, and nothing past it in 9 levels.
- **bar 3's novelty** (on the target's own inputs, the K candidates whose answers differ most from every instruction's
  and routine's): it prefers comparisons and subtractions in new placements, the addition at level 3, red on x + N,
  a chain at level 7. The multiply-accumulate row is learnable at level 4 only (its problems' one search over the
  primitives alone, as under rca), where four other candidates rank as more novel, so it is lost: no multiply and no
  reduction in 8 levels. On seed 1 it gets less far: the addition at levels 3-4, then no row, no chain, no multiply
  and no reduction in 6 levels (level 6 alone takes 1,884 s).
<!-- /ARM_BY_ARM -->

## 4. How hard each arm was tried

<!-- HOW_HARD -->
- **One run per arm and seed**, the untouched official manifest on one H100: seeds 74 and 1 for all 30 arms. 17 of
  the seed-1 runs were cut off by Modal's spend limit and run again from the start once it was raised (below). A
  certifying arm's run takes 1,000-1,600 s of wall time (practising everything, 3,573 s). A failing arm takes the
  whole 3,600 s: its climb, then the round searched again every 10 steps until the deadline. One failing run went
  past the budget: rule_lp on seed 1 trained 3,673 s. Its last search, building level 9's vocabulary, took 239 s and
  ended 73 s after the budget's end. It fails either way, but sp2's searches can run past the training deadline.
- **The control is exact.** `tools/modal_runner.py --mode compare` finds every field equal to sp2's official record
  (`records/runner/submission_stage3_sp2_official.json`): the training trace (201 points), the loss at every logged
  step, every rung and split, the freeze at step 887. The climb's 48 units equal the official run's line for line once
  their seconds are removed. Only the seconds differ (the round by 977 s against 1,029 s: another host). The seed-1
  control has no earlier H100 record to replay (sp2's seed 1 ran on a B200); it certifies, the round at step 630.
- **Seed 1 agrees with seed 74 on every arm but four.** All nine book arms not needed on seed 74 certify on seed 1,
  the round at step 627-637 (the control's 630), and so does rule_lpcc. The 15 arms needed on seed 74 fail on seed 1
  too. The four that differ (§3):
  - rule_rca and rule_lpc failed on seed 74 and certify on seed 1. Both take apart lpca, and both turn a sure thing
    into a chance: whether a random draw keeps the multiply-accumulate row, and whether the one practised placement
    is one the round can use.
  - no_pat_xdn and rule_all certified on seed 74 and fail on seed 1. The first needs a cost-5 program at the deep
    search's last level; the second needs the time to practise everything.
- **Host speed.** The climb is host-bound (INTEGRATION.md §3), and H100 hosts differ: on the probe and deep-probe
  units identical to the control's (same level, same line once seconds are removed), the arms' hosts ran at 0.71 to
  1.06 times the control's seconds on seed 74. `analyze.py` divides the climb's part of the round time by that factor
  when at least 10 units match; otherwise the raw seconds stand. The step of the round does not depend on the host:
  every certified arm found it at step 685-694 on seed 74 and 627-640 on seed 1 (the control: 688 and 630).
- **Did a failure run out of things to try, or out of time?** Unlike s3i's ablation, where every failed climb ended by
  its own rule, most failures here ran until the training deadline cut the climb (the deadline is 120 s before the
  budget's end; a level cut by it returns from its searches at once, so the climb then rushes through empty levels).
  The reason is sp2's rule: when no new concept is learnable, the next level poses the practised concepts' other
  placements, so the climb rarely runs dry. The exceptions are the four arms whose book can pose nothing past the pair
  (lt, red, add, the pair). Their climbs ran dry by their own rule on both seeds: at the level cap by 890-1,719 s, or
  after 75-80 s. So for each failure cut by the clock, the question is whether the missing piece was still within
  reach when the clock stopped. One failed run had a search stopped by sp2's 900 s per-search limit: rule_all on seed
  1, at level 4 (1,371 s with its vocabulary build).
  - mul, (x, o), x + o N: no, by construction. The round's search composes routine calls and straight-line moves, no
    loops, so it needs a routine that multiplies two numbers and one that reduces a 16-place number; with these out of
    the book, no law's routine does either.
  - a + b c and (o, x, d): no. Two levels that posed only concepts not yet practised (levels 5 and 7: 48 candidates,
    the multiply's in every placement among them) were probed and deep-probed in full, and nothing was learnable (on
    seed 1, one such level: level 5). The multiply needs an accumulating row as one instruction of its loop's body,
    and only a + b c poses one.
  - x + d N, on seed 1: not within reach of the search it had. The inline chain it needs is right with seed 1's
    routines, but levels 5, 7 and 8 searched every placement of x + o N to cost 5 in full without finding it (§3).
    Whether more levels, with a few more routines each, would have found it is not known.
  - `all`, on seed 1: out of time, not out of reach. The deadline cut the climb in level 4, the level whose round came
    on seed 74.
  - `rca`: no, on seed 74. The multiply-accumulate row's only search that reaches it (the primitives alone, once per
    root) was at level 3; afterwards every level's deep probes go to the same unsolved near miss (the quotient on
    x + d N).
  - `lpc`: not within 3 times the budget so far. The rerun of seed 74 at 10,800 s (H100) found a reduction at level 11
    (4,275 s), the divisor-shifting one seed 1 used (§3), but with its dividend in N, which no instruction writes; the
    round searches of levels 11-14 found nothing, and Modal's spend limit stopped the run in level 15 (6,229 s). Its
    last log lines are in [`partial/`](partial/); there is no record.
  - `lp`, `ct2a2`: no reduction in 10-12 levels of real work on seed 74, or in 8-9 on seed 1 (where both had the
    multiply and chains, chains that read or write X). SELF_PRACTICE.md's tiny core ran them longer (`lp` at 16
    probes: 10 levels, 3,976 s on a B200; `ct2a2`: 7 levels, 1,799 s), also without one.
  - `random_ok`, `lpu`, `ct1`, `nov`: no progress toward the missing pieces in their last levels, on either seed.
    random_ok never posed past the pair on seed 74 and never drew the row on seed 1. lpu had the multiply, but its
    chains use X. ct1 and nov lost or never reached the row.
- **The reduction check's weakness** (found by s3i's ablation: stage 3's check of a reduction can pass a repeated
  subtraction that is right only for small quotients) decides no verdict here. Every certified run's round was checked
  at full size by the evaluator. No failed arm practised both a multiply and a reduction, on either seed. Every
  multiply a failed arm practised is the control's own program, and so is every reduction on seed 74. On seed 1 four
  book arms practised another reduction, `CLA P[BRG.X T.add.pair.AN T.red.x+dn.AN]` with the modulus in N. It is
  right at every size [`reduce_check.py`](reduce_check.py) tries (W = 1 to 8, quotients up to 10^W), and those arms
  lack the multiply anyway. So each failure lacks a piece outright rather than holding a wrong one.
- **Cost.** Billed by Modal: $119.41 for the ablation's run (`sp-abl-s74`: seed 74, the 13 seed-1 records and the
  cut-off work), $3.08 and $49.00 for the books' runs (`sp-books-gen`, `sp-books-2`), all through 13:00 UTC; the last
  half hour before the stop adds about $14. About $185 in all, of a $300 soft budget.
- **The stop.** At 07:25 PDT (14:27 UTC) on Oct 24, Modal stopped every container in the workspace, every session's:
  the workspace reached its spend limit (about $3,000 for the month, set in Modal below the $5,000 this study was
  planned around). Seven seed-1 runs were cut off mid-run (no record), ten never started, and the 3x rerun of `lpc`
  was ended. The queued calls stayed PENDING but never resumed. When the limit was raised to $4,000 (14:40 PDT), they
  were ended first by stopping their apps, so none could run twice. Then the 17 seed-1 arms and the 3x rerun were run
  again from the start (runs `sp-abl-s1-rest` and `sp-abl-relaxed`, 18 H100s at once).
<!-- /HOW_HARD -->

## 5. A smaller book, and a generated one

### 5.1 The smallest book the arms allow (`book_min`)

Every law and every pattern whose own arm certified without it on seed 74 is dropped together, the x + d N pattern
too (on seed 74 it only cost time; its seed-1 run, which came later, fails, §3, but this book certifies on seed 1).
What is left is 5 laws (lt, red, add, mul, acc.mul) on 4 patterns (the pair, x + o N, (x, o) and
(o, x, d)): 6 concepts (lt and red on the pair, add on the pair, red on x + o N, mul on (x, o), acc.mul on
(o, x, d)), 84 candidates with the constructions' routines given, against sp2's 19 concepts and 250 candidates
(sp2's book has 20 cells; the quotient of x + N by N is always 1, so that one is never posed).
`check_variants.py`'s pose check: its pool is exactly the control's, restricted to those laws and patterns.

<!-- BOOK_MIN -->
**It certifies on the untouched official manifest on an H100** (`book_min_h100_seed74`): T = 64 on both ladders,
every rung 768/768, every split exact; the round by 571 s (step 689), frozen by 614 s, 799 s of training (the
control: 977 s, 1,022 s, 1,206 s). Its climb is the round's pieces and nothing else, one concept a level, each in its
tied-cleanest placements:

| level | practised | program |
|---|---|---|
| 0 | lt on the pair, in A against X, Y, N | `CLC P[SBB.Y INC]` |
| 1 | red on the pair (the link), the same three | `STA.S T.lt.pair.AY SEL.S` |
| 2 | add on the pair, the same three | `CLC P[ADC.Y INC]` |
| 3 | a + b c on (o, x, d) (the multiply-accumulate row), its four A placements | `LDH.Y CLK P[MUL.X INC]` |
| 4 | mul on (x, o) (the multiply), in X, Y both ways | `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]` |
| 5 | red on x + o N (the reduction), in X against A | `STA.Y P[BRG.X B[T.red.pair.AY]]`: the chain inline |
| round | | `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA`, sp2's round |

Each level poses 12-48 candidates on 12-48 roots (sp2: 36-177 on 12-96), so the climb costs about half of sp2's.

**And on all 8 seeds** (B200, the official manifest with `runtime.seeds` changed, the seed policy's finals): every one
certifies, exact everywhere, with sp2's round, the same six concepts at the same levels.

| seed | certified (T seen / unseen) | round: step (level), s | frozen by, s | training, s |
|---|---|---|---|---|
| 74 | yes (64 / 64) | 689 (L5), 538 | 579 | 763 |
| 1 | yes (64 / 64) | 631 (L5), 543 | 570 | 757 |
| 2 | yes (64 / 64) | 680 (L5), 579 | 614 | 834 |
| 3 | yes (64 / 64) | 599 (L5), 505 | 547 | 760 |
| 4 | yes (64 / 64) | 695 (L5), 504 | 543 | 723 |
| 5 | yes (64 / 64) | 635 (L5), 494 | 520 | 702 |
| 6 | yes (64 / 64) | 596 (L5), 443 | 524 | 732 |
| 7 | yes (64 / 64) | 680 (L5), 517 | 555 | 762 |

sp2 itself on the same 8 B200 seeds (SELF_PRACTICE.md §4.2) found its round by 650-1,775 s; the smallest book by 443-579
s. Its two hand-made-looking choices are then the only ones it has: which 5 laws and which 4 patterns, which are the
round's ingredients exactly. It shows what the result rests on; it is not a less designed book.
<!-- /BOOK_MIN -->

### 5.2 A book generated by a grammar (`book_gen`)

[`genbook.py`](genbook.py) replaces sp2's hand-written book (its `PATTERNS`, `LAW_BOOK`, `Draw` and `law_target`)
by one generated from a few rules. [`make_books.py`](make_books.py) splices it into a copy of sp2; nothing else in
the file changes.
- **Numbers**: the division law's form r + q b, on the two pairs the data says are ordered (r < b): x < N (the mined
  order) and a digit d < B (a digit's range). q is one, a digit, another row's x (o), or x itself. That gives x + N,
  x + d N, x + o N, x + x N, and the base forms d + x B (= 10 x + d) and d + o B.
- **Patterns**: the comparison pair (sp2's: x and N in both orders, with itself, with zero), each number with N, x
  with each count (x, d), (x, o), (x, x), and o, x with each count (o, x, d), (o, x, o), (o, x, x).
- **Operations**: the machine's binary arithmetic (lt, eq, min, max, red, div, add, sub, mul) and every composition
  f(a, g(b, c)) of its two value operations (+ and ×): a + (b + c), a + b c, a (b + c), a (b c).
- **Every operation on every pattern** of its arity. One set of target rules decides where a law speaks, the same for
  every law: a fact (x' <= N', equal digit strings), an identity (the division law, sub.inverse, sub.self,
  sub.identity, red.zero, add.identity, mul.zero), or a practised routine of the learner (its addition, repeated; its
  chain for the base form, as sp2's red.step). A law abstains where none applies, and where its largest value would
  not fit the registers (the bookkeeping sp2's width rule does by hand). So nothing lists which law goes with which
  pattern, and no composition is singled out: of the four, a + (b + c) and a + b c are the ones that fit.

Of the 13 patterns, 12 pose something (d + o B has no fact to bound it and abstains everywhere); with the
constructions' routines given, the book poses 554 candidates on 43 concepts, against sp2's 250 on 19: all 19 of sp2's
and 24 new ones (a few of those speak only on rare probes, such as lt on x + o N, which needs o = 0). It contains
sp2's book: on all 264 of sp2's (concept, binding) roots, at every probe and acceptance setting, its targets equal
sp2's, value for value under the same masks ([`check_books.py`](check_books.py), part 1; sp2's acc.mul and acc.add are
its add.mul and add.add). On every concept it poses at the three settings in every binding (41 on that check's draws,
22 of them new), every target it gives equals the law's value by Python's integers (12,407 probe targets; part 2, a
check only), and the addition's check passes true sums and fails wrong ones.

<!-- BOOK_GEN -->
**It certifies on the untouched official manifest on an H100** (`book_gen_h100_seed74`): T = 64 on both ladders,
every rung 768/768, every split exact; the round by 2,198 s (step 716, level 5), frozen by 2,228 s, 2,428 s of
training. That is 1,221 s later than sp2's control, inside the budget. Its climb, concept by concept:

| level | practised (sp2's concepts in plain type, the generated book's own in bold) | seconds |
|---|---|---|
| 0 | lt and sub on the pair | 33 |
| 1 | **div on the pair** (x div N is 0, or 1 where x = N), min and red on the pair | 62 |
| 2 | max, eq, add on the pair | 168 |
| 3 | a + b c on (o, x, d) (the multiply-accumulate row), **sub on x + o N**, red and sub on x + N | 569 |
| 4 | **a + b c on (o, x, x)** and **on (o, x, o)**, red on x + d N (the chain), mul on (x, d) | 596 |
| 5 | red on x + o N, **red on x + x N**, red on 10 x + d, mul on (x, o) | 647 |

The round it found is not sp2's: `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X T.red.x+xn.XN`. x goes to Y, A is cleared,
and the generated multiply-accumulate of two numbers, `T.add.mul.o,x,o.AXY` = `W[T.add.mul.o,x,d.AYX INC SHL.Y]` (A +
X Y: the row looped over Y's digits), gives 0 + x x; then the reduction it practised on the generated x + x N,
`T.red.x+xn.XN` = `LDA.N P[BRG.X T.red.x+dn.AN]` (long division with the modulus in N, starting from A = N, which is 0
mod N). The other three programs the tournament holds are the same up to registers. 54 routines and 56 merges at the
round (sp2's control: 47 and 34).

Why it is slower: from level 3 on its pool is about twice sp2's (421 candidates on 156 roots at level 3, against
156 on 84), so each level costs 570-650 s against sp2's 200-330 s. And one of level 3's four concept slots went to a
generated concept (sub on x + o N, cost 3); the chain (cost 2) lost its tie with red and sub on x + N, which come
first in the pool, and came a level later, and the reduction with it. Level 4's deep probes all went to a + b c on
(o, x, x), close (0.5) without a program there (sp2's deep probes take a closest concept whole, SELF_PRACTICE.md §5
item 10, and this one has 24 placements).

**And on all 8 seeds** (B200): every one certifies, exact everywhere, by one of three routes, none of them sp2's:

| seed | certified (T seen / unseen) | round: step (level), s | frozen by, s | training, s | the round |
|---|---|---|---|---|---|
| 74 | yes (64 / 64) | 716 (L5), 1,942 | 1,968 | 2,132 | `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X T.red.x+xn.XN` |
| 1 | yes (64 / 64) | 661 (L6), 1,992 | 2,058 | 2,250 | `T.mul.x,x.XN STA.X LDA.N T.red.x+on.XA` |
| 2 | yes (64 / 64) | 707 (L5), 2,084 | 2,110 | 2,299 | `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X LDA.N T.red.x+on.XA` |
| 3 | yes (64 / 64) | 625 (L5), 1,924 | 1,950 | 2,146 | `T.mul.x,x.XN STA.X LDA.N T.red.x+on.XA` |
| 4 | yes (64 / 64) | 721 (L5), 2,053 | 2,081 | 2,288 | `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X LDA.N T.red.x+on.XA` |
| 5 | yes (64 / 64) | 663 (L5), 1,744 | 1,785 | 1,973 | the same |
| 6 | yes (64 / 64) | 625 (L5), 2,078 | 2,110 | 2,306 | the same |
| 7 | yes (64 / 64) | 692 (L4), 1,675 | 1,722 | 1,913 | the same |

- **The square through the multiply-accumulate of two numbers** (six seeds): A cleared, then a + b c on (o, x, o) with
  x in both X and Y, so 0 + x x. The reduction is sp2's (red on x + o N) on five of them and the generated one (red on
  x + x N) on seed 74.
- **The square as its own concept** (seeds 1 and 3): mul on (x, x), the generated pattern "x with itself as the
  count". Its probes put x in both registers, so a program that squares X passes them: `T.mul.x,x.XN` = `LDA.X B6.7`.
  In the round N holds the modulus, and the routine never reads it.

The rounds come by 1,675-2,084 s on a B200 (sp2: 650-1,775 s), on the H100 by 2,198 s: slower than sp2's book on
every seed, inside the budget on every one.
<!-- /BOOK_GEN -->

## 6. What was not run, and why

- **More seeds for the four parts that differ by seed** (lpca's order and its tied placements, x + d N, choosing at
  all): two seeds show that each can go either way, not how often. A failing run takes the whole hour, about $4.30 on
  an H100.
- **Why the inline chain was not built on seed 1** (no_pat_xdn, §3). The program is right with seed 1's routines and
  cost 5 was searched in full, so the cause lies in which items the deep search's vocabulary held. Finding it means
  rebuilding that vocabulary with seed 1's routines, a run for Modal: the search is too big for a laptop.
- **Found in passing** (not fixed; no verdict rests on it): sp2's searches can run past the training deadline.
  rule_lp's seed-1 run trained 3,673 s of its 3,600, because its last search, building level 9's vocabulary, took
  239 s. A certifying run that did this would be over budget.
- **bar 3's `random`** (no probes; each random pick given a full practice search) has no counterpart: sp2's only
  searches are its probes, so a random pick the probes did not solve has no program to practise. `random_ok` (K
  learnable candidates at random) is the nearest arm.
- **Relaxed-budget reruns** were run only where a failure could plausibly be the clock's (§4): `lpc` on seed 74.
  Two seed-1 failures that came later were not rerun. One is `all`, a level short of where seed 74's round came, so
  more time would likely make it speed only on both seeds. The other is x + d N, where whether more levels would
  find the inline chain is not known. The other failures are structural, and §4 says why for each: the book cannot
  pose the piece, or no practised routine can make it, or the climb's own gate keeps its deep searches elsewhere.
  bar 3's own rules (`lp`, `ct2a2`, `all`, `lpu`) also had long tiny-core runs in SELF_PRACTICE.md §3.2: 7 to 10
  levels and up to 6,693 s on a B200 each, without the round.
- **Single cells of the book** (one law on one pattern, such as lt on x + d N alone) were not removed one at a time.
  The arms remove a whole law or a whole pattern; the smallest book (§5.1) removes cells by combining both.
- **The search sizes and the rest of sp2** (the probe and deep-probe budgets, K, the probes per setting, the round's
  search, the machine): outside this study, which takes apart only the book and the rule.

## 7. Reproduce

From `cpu_first/`, with the sotaku venv's Python (`PY=~/github/sotaku/venv/bin/python`). The local commands take
under a minute each and stay under 1 GB; the runs go through the shared runner (tools/README.md).

Build the variants and the books (both assert sp2's md5 and refuse to change a file that differs on disk), and check
every arm:

```bash
$PY integration/sp_ablation/make_variants.py
$PY integration/sp_ablation/make_books.py
$PY integration/sp_ablation/check_variants.py
$PY integration/sp_ablation/check_books.py
$PY integration/sp_ablation/make_variants.py --diff no_pat_xdn        # one arm's edits
```

Write a plan, check the bill (no launch at the stop or above: $3,750 since the limit went to $4,000), launch, fetch:

```bash
$PY integration/sp_ablation/make_plan.py stage1_s74 --seeds 74 --hardware h100 --arms control,no_law_eq,...
$PY integration/sp_ablation/spend.py 3750
$PY -m modal run --detach tools/modal_runner.py --mode plan --plan integration/sp_ablation/plans/stage1_s74.json \
  --out integration/sp_ablation/records --cap 8 --run sp-abl-s74
$PY tools/modal_runner.py --mode fetch --run sp-abl-s74 --out integration/sp_ablation/records --wait
```

The 17 seed-1 runs and the 3x rerun cut off by the stop went the same way:
[`plans/stage1_s1_rest.json`](plans/stage1_s1_rest.json) (exactly the 17, written with `make_plan.py --only-missing`)
as run `sp-abl-s1-rest` (`--cap 17`), and [`plans/relaxed.json`](plans/relaxed.json) as `sp-abl-relaxed`
(`--cap 1`). Both took `--force`, because their names were registered to `sp-abl-s74`, and only after its queued
calls had ended, so no job ran twice.

Read the records:

```bash
$PY integration/sp_ablation/analyze.py              # one line per run (host factor, corrected round time, pieces)
$PY integration/sp_ablation/analyze.py --results    # this file's results table, with each arm's verdict
$PY integration/sp_ablation/analyze.py --levels     # every level's searches and picks, per run
$PY integration/sp_ablation/analyze.py --both       # §2's table: every arm on seeds 74 and 1
$PY tools/modal_runner.py --mode compare integration/self_practice/records/runner/submission_stage3_sp2_official.json \
  integration/sp_ablation/records/control_h100_seed74.json
```

Check a reduction a run practised against the truth, with that run's own routines (seconds, under 1 GB):

```bash
$PY integration/sp_ablation/reduce_check.py integration/sp_ablation/records/no_law_mul_h100_seed1.json \
  "CLA P[BRG.X T.add.pair.AN T.red.x+dn.AN]" X N
$PY integration/sp_ablation/reduce_check.py integration/sp_ablation/records/no_pat_xdn_h100_seed1.json \
  "STA.Y P[BRG.X B[T.red.pair.AY]]" X A
```
