# One allocator for bar 3's search: consolidating the rules

<!-- PLAIN -->
## In plain words

**The question.**
- Bar 3's learner decides where its search goes, and what to practise, by a pile of rules that earlier studies added
  one at a time.
- The coordinator asked whether one general mechanism could replace them, keeping the bug fixes, with an ablation to
  show what still carries each win.
- The user had asked: "can't solve these fixes through ablation + general good mechanisms + scaling?"

**1. The one-score allocator, designed from theory first, kept 1 of 16 earlier wins.**
- The design (PREREG.md) gives every candidate one score: the bits it is expected to save the target, by the
  learner's own description-length score, per state of search.
- It spends each round's budget best first: deeper probes (1M, then 10M states) and looks (the target searched with
  the routine).
- It practises every routine whose measured gain beats its library cost, with no fixed count.
- It kept only 2x + 1, in round 0. The other 15 earlier wins ran 5 rounds without coming. The 5 formulas no rule had
  brought did not come either, as predicted.

**2. Why: it is the myopic policy, and it stopped.**
- A look at 30k states rarely finds the target: 6% of 8,720 first looks showed any gain. A routine's worth often shows
  only with another routine, or a round later.
- The allocator took a null look as the routine's worth, so nothing beat its cost. 59 of its 93 rounds practised
  nothing, and a round that practises nothing repeats.
- This is the known weakness of the single-step metareasoning policy: it tends to stop too early (Hay, Russell,
  Tolpin and Shimony, UAI 2012).
- Its budget also matched the old rules only in nominal budgets. Round 0 searched a median 0.38 of hardest first's
  states. It never probed at 1M about half of hardest first's round-0 picks (29 of 60) and over a quarter of anchored
  pairs' (17 of 60), so those were never solved.

**3. What carries the wins: the uniform probe and a fixed number of picks, together.** The ablation (two
pre-registered addenda) added the old rules back to the allocator, alone and together:

| arm | earlier wins that came |
|---|---|
| the allocator (cs1) | 1 of 16 |
| + the uniform 1M probe, picks by the threshold | 4 of 15 |
| + four picks a round, without the uniform probe: hardest first; four by value; anchored pairs | 2 of 15; 1 of 15; 1 of 7 |
| + hardest first and the uniform probe | 11 of 15 (11 of its 12), each at its old round or earlier |
| + anchored pairs and the uniform probe | **13 of 16 (13 of its 15), each at exactly its old round** |
| the same with the subgoal's tie-break off | the same 13, at the same rounds |
| + four picks by max(measured gain, prior), with the uniform probe | 4 of 16 |

- Either half alone brings little. Together, the old rules come back; the rebuilt rules mostly made the old round-0
  picks too.
- Both halves are general: neither names a formula. The count of four is a constant.

**4. What can go, and what cannot.**
- **One old rule carries almost everything:** anchored pairs with the uniform probe and four picks a round. It
  brought 13 of the 16 earlier wins with every patch off: the twin, the twin as a fifth pick, the deeper look, the
  big-x score, the rest's tie-break. None of these wins needed them; they were never in the configs that won these
  formulas.
- **The subgoal's tie-break does no work.** With it off, under anchored pairs, nothing changed.
- **10·x² + 1 still needs hardest first with the rest's tie-break,** the only patch with a win of its own.
- **The subgoal's separately budgeted deeper probe became a third probe rung, taken by value.** It did the subgoal's
  work on x³ and x² + 3: it gave x² mod N its 10M probe in round 1, and both came.
- **But by value the 10M probes also go to the subgoal score's table candidates,** at about 150 bits each. min(N∸x, x)
  and le(x², N) got solved at cost 7, became the costliest candidates, and took anchored pairs' anchors on x⁴ and
  10·x², and hardest first's picks on 10·x². Those are the rebuilt rules' other two losses (x⁴, and 10·x², which was
  undecided when stopped).
- **One score did not stand for both choosing rules.** Four picks by max(measured gain, prior) brought 4 of 16. Its
  prior includes the table score's 150 bits, so it practised the table candidates first.

**5. What this answers.**
- **Mostly yes.** The wins are carried by general mechanisms plus scale:
  - probe every candidate at a fixed 1M budget (the scaling part);
  - practise a fixed number of routines a round;
  - rank them by measured looks at single routines and at pairs anchored on the costliest (anchored pairs).
- **That is one of the old rules.** The pile around it can go, with two exceptions:
  - 10·x² + 1's tie-break;
  - some form of the subgoal's deeper probe. Its by-value replacement here did its work but cost x⁴, and perhaps
    10·x².
- **What did not work** is the study's attempt to put even that rule into one score. Bits per state with a threshold,
  measured myopically, vetoes the routines whose worth shows later.
- **The bug fixes stay.**
  - The row check is on throughout.
  - Bar 3's machine starts clean.
  - Its search lists every spelling, so stage 3's SPELL blind spot is not in it.
- **The composition harness was not tried:** the allocator did not go well, which was the coordinator's condition.
- **Cost about $49.5:** under the $50 cap, over the soft $40, because of the billing report's hour of lag (§10).

<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md (frozen 11:25 PDT, read-only, sha256 b2418869c80e9118d94cee0e2bcea1ee46f4118460d05a86b10bac0f3af37f50),
unchanged:

# One allocator for bar 3's search: pre-registration

Written 2026-10-26 11:25 PDT, before any run of this study. CONSOLIDATE.md §1 repeats this text unchanged.

## The question

Bar 3's learner decides where its search goes by a pile of rules, each added by a study:
- two fixed probes for every candidate (30k states, then 1M for the unsolved);
- the subgoal's deeper probe (10M);
- the subgoal's ties, and the twin's slot (picks), and the twin as a fifth pick (reach);
- the tie-break by the rest's score (picks);
- the deeper look at steps (reach);
- exactly four picks a round, ties in pool order;
- two choosing rules: hardest first (cost) and anchored pairs (looks, singles and pairs).

Can one general mechanism replace them, with the bug fixes kept, and keep the wins? Where it loses one, which old rule
was doing real work, and is that rule general?

## Theory, in brief

- **The learner's choice each round is metareasoning:** where to spend search, and what to keep, so that the target's
  description gets shorter. The principled allocation spends each next unit of computation where its expected value
  per unit cost is highest (Russell and Wefald, *Principles of metareasoning*, Artificial Intelligence 49, 1991).
- **Here the value is bits:** the learner's own description-length score (bar3/learner.mdl, Rissanen's MDL), the
  target's best program's tokens plus its exceptions.
- **The cost is states:** a breadth-first search's cost is its last level, and each level is a roughly constant factor
  larger (Levin's search; LARGE_NUMBERS.md §3.1). So the next depth of a search costs its next budget rung.
- **What a practised routine is worth** is the bits it saves the target, measured by a look (the target searched with
  it), less what it costs to add: its program's tokens, at the merge's library weight. That is the two-part code of
  MDL.
- **The old rules are proxies for that value:**
  - hardest first's cost is the bits a routine saves per use;
  - anchored pairs' looks measure the bits directly;
  - the subgoal's score estimates them before a problem is solved;
  - the fixed four picks and the probes' fixed budgets spend without asking.

  One score, bits per state, can stand for all of them.

## The allocator (ALLOC; `alloc.py`)

Per round, after the target probe and the pool, as before:
1. **The census:** every candidate's 30k-state probe (the old first probe, unchanged).
2. **The round's budget R,** matched to the old rules: 1M states for each candidate the census left unsolved, plus 10M,
   plus 30k for each learnable candidate. That is what the old second probe, the subgoal's deeper probe and single
   looks would spend.
3. **Actions, each with a cost in states:**
   - probe an unsolved candidate at its next rung: 1M, then 10M;
   - look at a learnable one: the target searched with its routine, at 30k, then at 1M.
4. **Values in bits, one estimate for every candidate** (tok = log₂ of the symbols, exc = the bits of one exception):
   - **a learnable candidate, looked:** its measured gain, mdl0 − mdl(its deepest look);
   - **learnable, not yet looked:** its prior, (cost − 1)·tok, the tokens its routine saves per use. If the subgoal
     score gives it more, that;
   - **unsolved:** the subgoal score in bits (a law exact: mdl0 − 3·tok; else the rows a table saves × exc). If none,
     (the census's last level − 1)·tok × its best fit's share of its probes;
   - **a duplicate's value passes to the learnable candidate it duplicates** (as the twin did).
5. **Best first:** repeatedly run the 32 actions with the highest value per state (ties: the cheaper action, then pool
   order), update the values from the results, until R is spent or no action has a positive value.
6. **Practice:** every learnable candidate whose measured gain exceeds its library cost (its program's tokens × tok ×
   0.5, the merge's library weight). No fixed count, and nothing practised when none passes. Then the merge and the
   target search, as before.

**Kept, as bug fixes:**
- the row check (ROWCHECK.md), on in every allocator arm;
- bar 3's machine already starts clean;
- its search already lists every spelling of each goal state, up to 200 programs a level, so SPELL's blind spot is
  not here.

**Replaced:**
- the uniform second probe and the subgoal's deeper probe (by rungs, by value);
- the four picks, pool-order ties, the subgoal's ties and pick, the twin and the twin as a fifth pick, and the rest's
  tie-break (by the gain threshold and values);
- the deeper look (by the look's rung);
- both rules (by one score).

Anchored pairs' pair looks are left out; the ablation adds them back if a win needs them.

## Runs

The allocator on bar 3's counting machine, each formula once, from scratch, at most 5 rounds, stopping when the target
comes, its sources and budgets those of the earlier win's config:
- **the held-out test's config** (`<rule>_deep_sub`'s sources): the ten held-out formulas, (x² + x), (x + 1)², 3·x², x⁴,
  (x³ + x), 2^x, Lucas, Tribonacci, floor(x²/N) and the triangular numbers; and the tracked formulas, x² mod N, x² + 1,
  x² − 1, x² + 3, 2x + 1, Fibonacci, x³ and 10·x²;
- **the two-source config** (newton, `_ns`): the triangular numbers and 10·x² + 1;
- **the digits config** (`_b_src`): binary x mod N.

Where the allocator loses an earlier win:
- the old rule that differs at the first round where the runs differ is added back alone, and the run repeated;
- that rule is named, and judged general or not.

Searches the earlier runs made are read from their stores. Money: soft $40, cap $50, `spend.py 4750` before each
launch, one spend watcher, local runs under a memory cap. If the money runs short, the earlier wins come first.

## Predictions

| formula | earlier (hardest first; anchored pairs) | the allocator | confidence |
|---|---|---|---|
| x² mod N, 2x + 1 | r0; r0 | comes, round 0 | high |
| x² + 1, x² − 1 | r1, r4; r0, r0 | comes, by round 2 | medium-high |
| x² + 3 | r1; r1 | comes | medium |
| (x² + x), (x + 1)², 3·x², x⁴ | r1; r0–1 | comes, by round 2 (x² mod N's deeper rung is taken by value) | medium |
| x³ | r1; r1 | comes | medium |
| 10·x² | r1; r4 | comes | low-medium |
| Fibonacci | r4; r1 | comes | medium |
| 2^x | no; r1 | comes (its cheap step's look gain is large) | medium |
| triangular, the two-source config | no; r1 | comes (the step's look gain) | medium |
| 10·x² + 1, the two-source config | r2 (with the rest's tie-break); no | comes | low |
| binary x mod N, the digits config | r3; r1 | comes | low-medium |
| (x³ + x), Lucas, Tribonacci, floor(x²/N), triangular (held-out config) | no; no | no | high |
| the allocator over all earlier wins | | keeps most; where it loses one, the reason is a round with too little practised (no gain beats its cost), and the rule doing real work is the fixed four picks | medium |

PREREG_ADDENDUM1.md (frozen 12:12 PDT, read-only, sha256
7b633911b395ea287ac5b1a50ed365dfcccbaa1f3975cf90f61fbd3a640c3a8f; its header's 12:20 was a time written ahead),
unchanged:

# Addendum 1: the ablation, pre-registered

Written 2026-10-26 12:20 PDT, after the pre-registered allocator's runs (tag cs1) and before any run below.
CONSOLIDATE.md records its hash.

## What cs1 showed (why these arms)

**Outcomes:**
- The allocator came for 2x + 1 (round 0) and for none of the other earlier wins.
- That covers x² mod N, x² ± 1, x² + 3, Fibonacci, x³, 10·x², (x + 1)², 3·x², x⁴, 2^x, the triangular numbers and
  10·x² + 1 (two-source config), and binary x mod N (digits config).
- x² + x was still running at writing (round 3, not come).
- The non-wins did not come, as predicted.

**Mechanisms, measured over its 93 rounds** (logs/cs1_mechanisms.txt, logs/cs1_oldpicks.txt):
1. **59 rounds practised nothing.** No measured gain beat its library cost, so the library did not change, and each
   later round repeated the one before.
2. **Of 8,720 first looks, 6% showed any gain.** A first look is the target searched at 30k states with one routine. A
   null look replaces the routine's prior, so that routine is never practised.

   A look this shallow rarely finds the target. A routine's worth can also show only with another routine, or a round
   later: hardest first's round-0 picks for x³, practised with no measured gain, brought x³ in round 1.
3. **Round 0 searched a median 0.38 of hardest first's states.**
   - R counts nominal budgets, while a search overshoots its nominal budget by finishing its last level.
   - So the allocator probed at 1M about a quarter of the candidates the old rule probed.
   - Most of the old rules' round-0 picks were among those it left unprobed (x² mod N: 0 of hardest first's 4 were
     learnable to it).

## Arms

Each arm is the cs1 allocator (alloc_deep, alloc_ns or alloc_b_src) with one old rule added back, or two. Each rule sits
behind its own flag, off in cs1.
- **`probe`:** the old uniform second probe. Every candidate the census left unsolved is probed at 1M, outside R. The
  picks are still the gain threshold's.
- **`lp`:** hardest first's picks in place of the threshold: the 4 costliest learnable candidates (ties: the subgoal's,
  then pool order). The probes are still by value.
- **`k4`:** the fixed count alone: the 4 learnable candidates of highest value (the measured gain, else the prior), with
  no threshold.
- **`lp_probe`:** `lp` and `probe` together. This is close to the old hardest first; the differences are the row check,
  and the third rung taken by value in place of the subgoal's deeper probe.
- **`ct2a2`:** anchored pairs' picks, run by the old rule's own code on the allocator's classification. That code looks
  at every learnable candidate, adds pair looks anchored on the 2 costliest, and picks four.
- **`ct2a2_probe`:** `ct2a2` with the uniform probe. This is close to the old anchored pairs.

**Formulas:**
- `probe`, `lp`, `k4` and `lp_probe` run on every earlier win the allocator lost:
  - x² mod N, x² + 1, x² − 1, x² + 3, Fibonacci, x³, 10·x², (x + 1)², 3·x², x⁴, x² + x and 2^x (alloc_deep);
  - the triangular numbers and 10·x² + 1 (alloc_ns);
  - binary x mod N (alloc_b_src).
- `ct2a2` and `ct2a2_probe` run where anchored pairs won alone or first: 2^x, the triangular numbers, x² − 1, x² + 1,
  x² + x, Fibonacci and binary x mod N.
- That makes 74 runs, in 15 groups of one formula each, read from the cs1 group's store.
- If cs1 brings x² + x after all, its arms count as a check, not as an ablation of a lost win.

## Predictions

| arm | lost wins that come | confidence |
|---|---|---|
| `probe` | at most 4 of 15: the threshold still blocks, since most looks stay null | medium |
| `lp` | 5 to 9 of hardest first's 12; where its picks needed the 1M probe, they differ | low-medium |
| `k4` | 4 to 8 of 15: no stall; an unlooked costly routine's prior outranks most measured gains, so it often picks as `lp` does | low |
| `lp_probe` | at least 9 of hardest first's 12 (x² mod N, x² ± 1, x² + 3, Fibonacci, x³, 10·x², (x + 1)², 3·x², x⁴, x² + x, binary x mod N); not 2^x, the triangular numbers or 10·x² + 1, which hardest first never brought without a patch | medium-high |
| `ct2a2_probe` | at least 5 of its 7 | medium |
| `ct2a2` | 1 to 4 of 7 | low |

## How the outcome is read

These rules are written before the runs:
- **If `lp_probe` and `ct2a2_probe` bring most of the lost wins and `probe` does not,** then what carries the wins is
  practising a fixed number of routines every round, ranked by cost or by looks. The threshold is what loses them.
- **If `lp` brings about as many as `lp_probe`,** hardest first does not need the uniform probe.
- **If `k4` brings about as many as `lp`,** the fixed count is the working part, not the ranking by cost.

**What counts as general:** a rule is general when it names no formula, problem or source, and has no constant set for a
formula.
- Hardest first is general. Cost is a property of any routine.
- The fixed count of four is general, but a constant.
- The uniform probe is general. It is a budget rule.
- Anchored pairs are general.

## Money

- cs1 cost about $5. These 74 runs are estimated at $15 to $20.
- The limits stand: soft $40, cap $50 overall, `spend.py 4750` before each launch, and one watcher for all apps.
- If money runs short, `lp_probe` and `ct2a2_probe` come first (the old rules, rebuilt), then `probe`, `lp`, `k4` and
  `ct2a2`.

PREREG_ADDENDUM2.md (frozen 12:43 PDT, read-only, sha256
bf525179ae871dfa7ecb8a82f89dc8b14ee0e8b0fc02ba1dc3e56025582fabfc; its header's 12:45 was a time written ahead),
unchanged:

# Addendum 2: one rule for all the wins, pre-registered

Written 2026-10-26 12:45 PDT, while addendum 1's runs (tag ab1) were under way, before any run below. CONSOLIDATE.md
records its hash.

## What ab1 had shown at writing (20 of its 74 runs ended)

**The rebuilt old rules bring their old wins back:**
- `lp_probe` (hardest first with the uniform probe) had brought 7:
  - x² mod N, r0;
  - x² − 1, x² + 3, x³, x² + x, (x + 1)², 3·x², each r1.
- `ct2a2_probe` (anchored pairs with the uniform probe) had brought 4: x² + 1 r0, x² − 1 r0, x² + x r0, Fibonacci r1.
- In 5 of the 6 runs with a round 0 so far, `ct2a2_probe` made all four of the old anchored pairs' round-0 picks
  (x² − 1: 3 of 4). `lp_probe` made 3 or 4 of hardest first's, except on 10·x²: 1 of 4.
  - There the allocator's third rung, taken by value, solved three more cost-7 candidates that the old learner left
    unsolved (it gave 10M only to the subgoal).
  - Hardest first took those in place of the constant 10 in three registers.

**Either half alone brings little:**
- `probe` (the uniform probe, picks by the threshold) had brought 3: x² mod N, x² − 1, x² + 3.
- `lp` (hardest first without the uniform probe) had brought none.
- `k4` (four by value) and `ct2a2` (anchored pairs without the uniform probe) had brought only x² − 1.

## Questions

1. **Does one old choosing rule carry every earlier win?**
   - By the earlier records, anchored pairs alone brought 15 of the 16 wins, all but 10·x² + 1.
   - ab1 runs `ct2a2_probe` on 7 of them. Here it runs on the other 9.
2. **Does the subgoal's tie-break do work?**
   - It is the last of the coordinator's listed rules still inside the rebuilt rules.
   - `ct2a2_probe_noties` turns it off. The subgoal's deeper probe stays, as the allocator's third rung.
3. **Can one score stand for both choosing rules, given a fixed count and the uniform probe?**
   - `kmax_probe` practises the four learnable candidates of highest max(measured gain, prior). The prior is the
     allocator's: (cost − 1)·tok, the bits the routine saves per use, or the subgoal score's bits when more.
   - So a null look cannot erase a routine's prior. That is the optimistic estimate: a shallow look that finds
     something shows a gain, and one that finds nothing shows little.
   - It was designed after cs1 and the part of ab1 above.
   - `k4_probe` (four by the allocator's own value) is not run: under the uniform probe nearly every learnable
     candidate is looked at, so its ranking among null looks would fall to pool order. The money goes to `kmax_probe`.

## Arms and formulas (40 runs, in 16 groups of one formula each)

- **`ct2a2_probe`** on 9 formulas: x² mod N, 2x + 1, x² + 3, x³, 10·x², (x + 1)², 3·x², x⁴ (alloc_deep), and 10·x² + 1
  (alloc_ns).
- **`kmax_probe`** on all 16 earlier wins.
- **`ct2a2_probe_noties`** on the 15 earlier wins anchored pairs ever brought: all but 10·x² + 1.
- **Stores:** each group reads its ab1 group's store, or cs1's for 2x + 1.

## Predictions

| arm | earlier wins that come | confidence |
|---|---|---|
| `ct2a2_probe`, the 9 | at least 6 of anchored pairs' 8: x² mod N r0, 2x + 1 r0, x² + 3 r1, x³ r1, (x + 1)² r1, 3·x² r1, x⁴ r1, 10·x² (r4 before, so the least sure); not 10·x² + 1 | medium-high |
| `ct2a2_probe_noties` | the same as `ct2a2_probe`, but for at most 2 | medium |
| `kmax_probe` | 6 to 11 of 16. The subgoal score's bits outrank most measured gains and costs, so it practises the subgoal's candidates first | low |

## How the outcome is read

- **If `ct2a2_probe` brings all of anchored pairs' wins:** one old rule carries every earlier win but 10·x² + 1. That
  rule is anchored pairs, with the uniform 1M probe and four picks a round. Hardest first, the subgoal's separately
  budgeted deeper probe, the twin, the twin as a fifth pick and the deeper look can all go. 10·x² + 1 needs hardest
  first with the rest's tie-break.
- **If `ct2a2_probe_noties` brings the same:** the subgoal's tie-break can go too.
- **If `kmax_probe` brings about as many as `ct2a2_probe`:** one score can stand for both choosing rules, given a fixed
  count and the uniform probe.

## Money

- **So far:** cs1 cost about $5. ab1 is estimated at about $19, from its states: about $0.05 per billion states, by
  cs1. These 40 runs are estimated at about $12.
- **The limits stand:** soft $40, cap $50, `spend.py 4750` before each launch, one watcher for every live app.
- **If money runs short,** the 9 `ct2a2_probe` runs come first, then `kmax_probe`, then `ct2a2_probe_noties`.

### 2. Theory and design

#### 2.0 Theory, and what the runs say about it

**The design (PREREG.md):** the learner's choice of where to search and what to practise is metareasoning. Each next
unit of computation goes where its expected value per unit cost is highest (Russell and Wefald, *Principles of
metareasoning*, Artificial Intelligence 49, 1991). Here the value is measured in bits by the learner's description
length score (MDL), and the cost in states searched. A practised routine is worth the bits it saves the target, less
its library cost: MDL's two-part code.

**What that theory already warns of:**
- **The allocator is the *myopic* policy.** A routine's value is estimated from one computation (one look), as if no
  other routine and no later round could change it. Russell and Wefald's own approximation is the single-step
  assumption: value a computation as if it were the last one taken.
- **The myopic policy tends to stop too early.** Hay, Russell, Tolpin and Shimony (*Selecting computations: theory and
  applications*, UAI 2012, arXiv:1207.5879) say so, because changing one's mind about which action to take often
  needs more than one computation.
- **Myopic estimates often badly underestimate the value of information** (Tolpin and Shimony's semi-myopic work,
  cited there).

**cs1 is that failure, in this setting:**
- A routine's worth often shows only with another routine (anchored pairs' pair looks exist for this), or one round
  later, once it is in the library. One 30k look sees neither.
- The gain threshold then turns an underestimate into a veto. Nothing is practised, nothing changes, and the run stops
  in effect while its rounds go on.

**What the old rules did instead:**
- Practising a fixed number of routines every round is forced exploration. It is the standard remedy when a greedy
  policy's estimates are biased low: an arm judged by one weak, low-biased sample is never tried again otherwise.
- Hardest first's prior, a routine's cost, is the bits it saves per use. It is an estimate that needs no look, so it
  is not biased by the look's shallowness.

#### 2.1 The allocator (`alloc.py`)

`probe_and_choose` replaces, when `cfg["alloc"]` is on, the learner's two fixed probes, the subgoal's deeper probe, the
choosing rule and its patches. It is PREREG.md's design; the implementation adds these details:
- **Every search is an old job:** the old kinds (`probe_a`, `probe_b`, `probe_deep`, `look`) at the old budgets (30k;
  1M; the subgoal's 10M; 30k, then reach's 1M deeper look), so the earlier runs' stores serve them. Round 0's census
  was read from the store in every run.
- **Groups:** learnable candidates whose best programs share a signature are one group. The first in pool order is
  looked at, valued at the group's best value.
- **Batches:** up to 32 actions, in order of value per state (ties: the cheaper, then pool order), while their nominal
  budgets fit in what is left of R (at least one action).
- **Spending** counts the states each search visited against R. R is counted in nominal budgets (§5.3 says what this
  mismatch did).
- **The classification at the end** is the old learner's alias and duplicate checks with the row check, in pool order
  on the final results. A duplicate's measured gain passes to its original, as the twin did.
- **The record** of each round carries `alloc`: the budget, the states spent, each search (its kind, candidate, value,
  states, cost found, gain) and the picks.

#### 2.2 The learner copies (`make_learners.py`)

`learn_newton.py` and `learn_digits.py` are written from bar3/scale/reach's two copies (the learner frozen at 97f30b9,
with every earlier study's flags), with 4 asserted replacements marked "consolidate":
- the import;
- the allocator in place of the probes;
- its picks in place of the choosing step;
- its record in the round's record.

With `cfg["alloc"]` off each copy is reach's (§3). The ablation's rebuild changed one line, so that with
`cfg["alloc_oldpick"]` the old choosing rule runs on the allocator's classification.

#### 2.3 The configs (`configs.py`)

Each earlier win's config, with:
- the allocator in place of its rule;
- the row check on (a bug fix);
- every old patch to the picks off: the rest's tie-break, the twin, the twin as a fifth pick, the deeper look, the
  big-x score.

The subgoal score stays, as one of the allocator's values, and its deeper probe's budget is the allocator's third probe
rung. The configs are:
- `alloc_deep`: `lp_clean_deep_sub`'s sources and budgets, the held-out test's frozen config;
- `alloc_ns`: `lp_clean_deep_sub_ns`, the two-source config;
- `alloc_b_src`: `lp_clean_b_src`, the digits config.

The ablation's configs add the old rules back by flags (PREREG_ADDENDUM1.md):
- `_probe`: `alloc_probe_all`, the uniform second probe, outside R;
- `_lp`: `alloc_pick="lp"`, hardest first's four;
- `_k4`: `alloc_pick="k"`, four by value;
- `_lp_probe`;
- `_ct2a2`: `alloc_oldpick` with `rule="ct2"`, `anchor_cost=2`, the old anchored pairs' own code;
- `_ct2a2_probe`.

Addendum 2's configs (PREREG_ADDENDUM2.md):
- `_ct2a2_probe_noties`: the subgoal's ties off (`subgoal=dict(deep=DEEP, ties=False)`);
- `_kmax_probe`: `alloc_pick="kmax"`, four by max(measured gain, prior), with the uniform probe.
- `_k4_probe` is defined but was not run.

#### 2.4 The runner (`modal_consolidate.py`)

bar3/scale/reach/modal_reach.py's `group`, on this study's app and Volume (`bar3-consolidate`):
- one container per group (16 cores, 64 GB), a thread per run, the searches on a process pool;
- each group reads and extends its own copy of a store on the Volume.

### 3. The tests before the arms

- **The tiny test** (`tiny.py`; `logs/tiny_h1.txt`): `alloc_deep` on x² mod N at the learner's TINY budgets, 2 rounds,
  locally under a 10 GB cap. It ran the code path end to end:
  - round 0: a 297-candidate census, a 1.3M budget, 64 searches, two picks;
  - round 1: no picks, which was the first sign of the stall (§5.2).
- **The replay check** (`check.py off`; `logs/check_off.txt`): with `cfg["alloc"]` off, the copies reproduce
  `lp_clean_deep_sub_h1_s1` and `lp_clean_deep_sub_rho_s1` (bar3/scale/subgoal/records) exactly, from their stores.
  This held before cs1 and again after the ablation's rebuild.
- **The ablation's flags** (`logs/tiny_flags.txt`):
  - with every flag off, the tiny record is the same after the flags were added and after the rebuild (apart from the
    store's `cached` marks);
  - each flag on runs its rule, addendum 2's (`kmax`, the ties off) included;
  - after `kmax` was added, the flags-off tiny record was again unchanged.

### 4. The runs (`records/*_cs1.jsonl`, `*_ab1.jsonl`, `*_ab2.jsonl`; `logs/launch_*.txt`)

- **cs1, the pre-registered allocator:** 21 runs in 7 groups, launched 11:40–11:44 PDT. They read a store joined from
  18 earlier stores: 93,764 distinct searches from the held-out, subgoal, picks, digits and reach studies. It is on the
  Volume as `v1/STORE.jsonl`, 87 MB, not in the repo.
  - A first launch loop failed before spawning anything except binary x mod N's group: zsh does not split an unquoted
    variable, so the targets were never passed. It was relaunched with explicit arguments (`logs/launch_cs1.txt`).
- **ab1, the ablation:** 74 runs in 15 groups (one formula each), launched 12:13–12:17 PDT. Each read its cs1 group's
  store.
- **ab2, addendum 2:** 40 runs in 16 groups (one formula each), launched 12:44–12:48 PDT. Each read its ab1 group's
  store, or cs1's for 2x + 1. Addendum 2 was frozen at 12:43 PDT, after 20 of ab1's runs had ended
  (`logs/ab1_at_addendum2.txt`). Its header says 12:45; its sha256 is
  bf525179ae871dfa7ecb8a82f89dc8b14ee0e8b0fc02ba1dc3e56025582fabfc (`logs/addendum2_sha256.txt`).
- **Addendum 1** was frozen read-only at 12:12 PDT, before the first ab1 launch. Its header says 12:20, a time written
  ahead; the file's own time is 12:12. Its sha256 is
  7b633911b395ea287ac5b1a50ed365dfcccbaa1f3975cf90f61fbd3a640c3a8f (`logs/addendum1_sha256.txt`).
- **PREREG.md** was frozen at 11:25 PDT, sha256 b2418869c80e9118d94cee0e2bcea1ee46f4118460d05a86b10bac0f3af37f50.
- **Preemptions:** several groups were preempted and replayed from their stores (the runner retries once). The replay
  rewrites a run's record from the start, and its rounds are read back from the store.
- **Two stops cut runs short** (§10): 9 low-priority runs by hand at 13:46 PDT, and 21 runs by the spend watcher at
  14:04 PDT. A cut run counts as not come by the rounds it ran.

### 5. The allocator's outcomes (cs1; `compare.py`, `logs/cs1_mechanisms.txt`, `logs/cs1_oldpicks.txt`)

#### 5.1 What came

The allocator kept one earlier win of 16: 2x + 1, in round 0, as before. The other 15 ran 5 rounds and did not come,
and neither did the 5 formulas no rule had brought.

| formula | config | the allocator | hardest first | anchored pairs | routines practised, by round |
|---|---|---|---|---|---|
| 2x + 1 | deep | **r0** | r0 | r0 | 10 |
| x² mod N | deep | no | r0 | r0 | 1, 1, 0, 0, 0 |
| x² + 1 | deep | no | r1 | r0 | 13, 1, 0, 0, 0 |
| x² − 1 | deep | no | r4 | r0 | 1, 0, 0, 0, 0 |
| x² + 3 | deep | no | r1 | r1 | 1, 1, 0, 0, 0 |
| Fibonacci | deep | no | r4 | r1 | 2, 0, 0, 0, 0 |
| x³ | deep | no | r1 | r1 | 0, 0, 0, 0, 0 |
| 10·x² | deep | no | r1 | r4 | 2, 0, 0, 0, 0 |
| x² + x | deep | no | r1 | r0 | 3, 2, 1, 0, 0 |
| (x + 1)² | deep | no | r1 | r1 | 1, 1, 0, 0, 0 |
| 3·x² | deep | no | r1 | r1 | 6, 2, 0, 0, 0 |
| x⁴ | deep | no | r1 | r1 | 1, 1, 0, 0, 0 |
| 2^x | deep | no | no | r1 | 6, 2, 0, 0, 0 |
| triangular | two sources | no | no | r1 | 6, 3, 2, 0, 0 |
| 10·x² + 1 | two sources | no | no (r2 with the rest's tie-break) | no | 2, 0, 0, 0, 0 |
| binary x mod N | digits | no | r3 | r1 | 2, 2, 0, 0, 0 |
| x³ + x, Lucas, Tribonacci, floor(x²/N), triangular | deep | no | no | no | x³ + x: 18, then 0 |

#### 5.2 Why: three mechanisms, measured

1. **A round with nothing practised repeats.**
   - 59 of its 93 rounds practised nothing: no learnable candidate's measured gain beat its library cost.
   - The library then stays as it was, so the next round poses the same candidates (bar the dreams' draws), measures
     the same gains, and practises nothing again.
   - Every run's practice dried up by round 3. x³'s never started: 0 routines in all 5 rounds.
2. **A shallow null look vetoes a routine.**
   - Of 8,720 first looks (the target searched at 30k states with one routine), 6% showed any gain. The rest showed
     none, mostly just the cost of one more symbol (about −0.03 bits).
   - A look replaces the routine's prior with what it measured, so a null look leaves the routine worth nothing to
     the allocator: it is never looked at deeper, and never practised.
   - But a routine's worth often shows only with another routine, or a round later.
     - Hardest first's round-0 picks for x³ were four routines of cost 6. Two of them the allocator had (both learnable,
       never looked at before the budget ran out); the other two it never probed.
     - Hardest first practised all four with no measured gain, and x³ came in round 1.
     - For 10·x², hardest first's picks included the constant 10 in three registers (cost 7). The allocator's look
       measured −0.13 bits for it, and 10·x² never came.
3. **R matched the old rules in nominal budgets, not in states.**
   - A search finishes the level that crosses its budget, so a 1M probe visits several million states. R counted 1M.
   - Round 0 searched a median 0.38 of hardest first's states (0.31–0.54).
   - The allocator probed at 1M about a quarter of the candidates hardest first probed (x² mod N: 41 of 170), the
     ones the subgoal score valued most. A table-scored subgoal is worth its rows × one exception's bits, about 150
     bits, so these outbid everything else.
   - Of the old rules' round-0 picks, the allocator never probed at 1M 29 of hardest first's 60 and 17 of anchored
     pairs' 60 (`logs/cs1_oldpicks.txt`). x² mod N and binary x mod N had all 4 of hardest first's unprobed.
   - Addendum 1's motivation says "most"; the counts are these.

### 6. The ablation (ab1, ab2; `table.py`, `logs/table_final.txt`)

#### 6.1 What came

Each arm is the allocator with old rules added back. A cell is the round the target came, or "no" after 5 rounds; "no
(k†)" is a run stopped after k rounds. Two stops cut runs short:
- 13:46 PDT, by hand, under the addenda's money rule: the low-priority runs left on x² + 1, 2^x, the triangular numbers
  and 10·x² + 1;
- 14:04 PDT, by the spend watcher at its $44 total (§10).

"·" is an arm not run on that formula.

| formula | allocator | `probe` | `lp` | `k4` | `ct2a2` | `lp_probe` | `ct2a2_probe` | `ct2a2_probe` ties off | `kmax_probe` | hardest first before | anchored pairs before |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x² mod N | no | **r0** | **r3** | no | · | **r0** | **r0** | **r0** | no (4†) | r0 | r0 |
| 2x + 1 | **r0** | · | · | · | · | · | **r0** | **r0** | **r0** | r0 | r0 |
| x² + 1 | no | **r2** | no (4†) | no (4†) | no | **r1** | **r0** | **r0** | no (3†) | r1 | r0 |
| x² − 1 | no | **r1** | no | **r1** | **r1** | **r1** | **r0** | **r0** | no (3†) | r4 | r0 |
| x² + 3 | no | **r1** | no | no | · | **r1** | **r1** | **r1** | no (3†) | r1 | r1 |
| Fibonacci | no | no | no | no | no | **r4** | **r1** | **r1** | no | r4 | r1 |
| x³ | no | no | no | no | · | **r1** | **r1** | **r1** | no (2†) | r1 | r1 |
| 10·x² | no | no | no | no | · | no (4†) | no (4†) | no (4†) | no (3†) | r1 | r4 |
| x² + x | no | no | no | no | no | **r1** | **r0** | **r0** | no (3†) | r1 | r0 |
| (x + 1)² | no | no | no | no | · | **r1** | **r1** | **r1** | **r1** | r1 | r1 |
| 3·x² | no | no | no | no | · | **r1** | **r1** | **r1** | no (4†) | r1 | r1 |
| x⁴ | no | no | no | no | · | **r1** | no (4†) | no (4†) | no (3†) | r1 | r1 |
| 2^x | no | no (3†) | **r4** | no | no | no | **r1** | **r1** | **r1** | no | r1 |
| triangular (two sources) | no | no (2†) | no (4†) | no (4†) | no (4†) | no (3†) | **r1** | **r1** | no (3†) | no | r1 |
| 10·x² + 1 (two sources) | no | no | no | no | · | no (3†) | no (3†) | · | no (3†) | no (r2 with the rest's tie-break) | no |
| binary x mod N (digits) | no | no (1†) | no (2†) | no (3†) | no (3†) | **r3** | **r1** | **r1** | **r3** | r3 | r1 |
| **came** | 1 of 16 | 4 of 15 | 2 of 15 | 1 of 15 | 1 of 7 | 11 of 15 | 13 of 16 | 13 of 15 | 4 of 16 | 13 | 15 |

#### 6.2 What it says

1. **Either half alone brings little.**
   - The uniform 1M probe with the threshold's picks (`probe`) brought 4. The threshold still practised nothing in 29
     of its 54 rounds (x³: nothing in all 5).
   - A fixed count of four without the uniform probe brought 1 or 2: `lp` (hardest first) 2, `k4` (four by value) 1,
     `ct2a2` (anchored pairs) 1.
2. **Together they bring the old wins back, at the old rounds.**
   - Hardest first with the uniform probe (`lp_probe`) brought 11 of its 12 lost wins, each at its old round or
     earlier (x² − 1 in round 1, not 4).
   - Anchored pairs with the uniform probe (`ct2a2_probe`) brought 13 of its 15, each at exactly its old round.
   - The rebuilt rules made the old round-0 picks: anchored pairs all four in 10 of its 16 formulas and 3 of 4 in 3
     more; hardest first 3 or 4 of 4 in 14 of its 15.
   - They searched more for it: anchored pairs 33.9B states over its 13 shared wins against the old 23.8B, hardest
     first 56.0B over its 11 against 52.2B. That is the allocator's extra looks and third-rung probes.
3. **The subgoal's tie-break does no work here.** With it off, anchored pairs brought the same 13 wins at the same
   rounds, with the same picks.
4. **One old rule covers almost everything:** anchored pairs with the uniform probe and four picks a round.
   - Its 13 wins and hardest first's x⁴ make 14 of the 16 earlier wins between the two rebuilt rules.
   - The two left are 10·x² (both rules cut after 4 rounds; before, r1 and r4) and 10·x² + 1 (only ever with hardest
     first and the rest's tie-break).
   - Every patch was off: the twin, the twin as a fifth pick, the deeper look, the big-x score, the rest's tie-break.
     None of the wins here needed them; they were never in the configs that first won these formulas.
5. **The third rung's interaction.**
   - The subgoal's separately budgeted 10M probe became the allocator's third rung, taken by value.
   - The subgoal score values a table candidate at its rows × one exception's bits, about 150 bits, so the 10M probes
     went to those. min(N∸x, x) and le(x², N) were solved at cost 7, which the old learner left unsolved.
   - Solved, they became the costliest candidates. They took hardest first's picks on 10·x²: 1 of the old 4, in place
     of the constant 10 in three registers.
   - They took anchored pairs' anchors on x⁴ (min(N∸x, x) and mod(x, N), in place of mod(x, N) and lt(x + 1, N)) and
     on 10·x² (min(N∸x, x) and le(x², N), in place of the constant 10 in A and in T).
   - Those are the rebuilt rules' only lost earlier wins outside 10·x² + 1.
   - On x³ and x² + 3, the subgoal study found the deeper probe necessary. There the third rung did its job: in round
     1 it gave the subgoal (x² mod N) its 10M probe, which solved it at cost 6; it was practised, and the target came
     that round, under both rebuilt rules.
6. **One score did not stand for both choosing rules.**
   - Four picks a round by max(measured gain, prior), with the uniform probe (`kmax_probe`), brought 4 of 16: 2x + 1
     r0, (x + 1)² r1, 2^x r1, binary x mod N r3.
   - Its prior includes the subgoal score's bits, which outrank both costs and measured gains. So it practised the table
     candidates first (on x² mod N: mul(N∸x, x), min(N∸x, x), max(N∸x, x), mul(x − 1, N)).
   - 11 of its runs were cut after 2 to 4 rounds without coming. On all of those formulas but 10·x² and 10·x² + 1, the
     rebuilt rules had come by round 1.

**Which old rule did real work, and is it general:**

| rule | real work? | general? |
|---|---|---|
| the uniform second probe (1M for every candidate the census leaves unsolved) | yes: without it, any choosing rule brings 1–2 | yes: a budget rule, no formula named |
| a fixed count of four picks a round | yes: without it (the threshold), 4 with the uniform probe, 1 without | yes, with a constant |
| anchored pairs' ranking (looks at single routines and pairs anchored on the two costliest) | yes: 13 of 15, the most of any rule | yes |
| hardest first's ranking (cost) | yes, where anchored pairs' anchors went astray (x⁴); otherwise covered by anchored pairs | yes |
| the subgoal's tie-break | no: off, the same 13 | — |
| the subgoal's separately budgeted deeper probe | partly: a third rung by value did its work on x³ and x² + 3, but over-solved on x⁴ and 10·x² | as a rung, yes; its value (the table score's ~150 bits) is the problem |
| the twin, the twin as a fifth pick, the deeper look, the big-x score | not for any of these wins | — |
| the rest's tie-break | only for 10·x² + 1 | — |

### 7. Predictions against outcomes

#### 7.1 PREREG.md (the allocator, cs1)

| formula | predicted | came | right? |
|---|---|---|---|
| x² mod N, 2x + 1 | comes, round 0 (high) | 2x + 1 r0; x² mod N no | half |
| x² + 1, x² − 1 | comes, by round 2 (medium-high) | no, no | no |
| x² + 3 | comes (medium) | no | no |
| (x² + x), (x + 1)², 3·x², x⁴ | comes, by round 2 (medium) | no ×4 | no |
| x³ | comes (medium) | no; nothing practised in any round | no |
| 10·x² | comes (low-medium) | no | no |
| Fibonacci | comes (medium) | no | no |
| 2^x | comes (medium) | no | no |
| triangular, two sources | comes (medium) | no | no |
| 10·x² + 1, two sources | comes (low) | no | no |
| binary x mod N, digits | comes (low-medium) | no | no |
| x³ + x, Lucas, Tribonacci, floor(x²/N), triangular (deep) | no (high) | no ×5 | yes |
| the allocator over all earlier wins | keeps most; where it loses one, the reason is a round with too little practised (no gain beats its cost), and the rule doing real work is the fixed four picks (medium) | kept 1 of 16. The reason is as predicted, rounds with nothing practised (59 of 93), but far more often than predicted. The rules doing real work: §6 | the mechanism yes, its extent no |

#### 7.2 Addendum 1 (the ablation, ab1)

| arm | predicted | came | right? |
|---|---|---|---|
| `probe` | at most 4 of 15 (medium) | 4: x² mod N r0, x² + 1 r2, x² − 1 r1, x² + 3 r1. 2^x, the triangular numbers and binary x mod N were stopped after 3, 2 and 1 rounds | yes |
| `lp` | 5 to 9 of hardest first's 12 (low-medium) | 1 of the 12 (x² mod N r3), and 2^x r4, which hardest first never brought | no |
| `k4` | 4 to 8 of 15 (low) | 1: x² − 1 r1 | no |
| `lp_probe` | at least 9 of hardest first's 12; not 2^x, triangular or 10·x² + 1 (medium-high) | 11 of the 12, each at its old round or earlier. 10·x² had not come when stopped after 4 rounds (before, r1). Not 2^x (5 rounds); not triangular or 10·x² + 1 (stopped after 3) | yes |
| `ct2a2_probe` | at least 5 of its 7 (medium) | 7 of 7, each at its old round | yes |
| `ct2a2` | 1 to 4 of 7 (low) | 1: x² − 1 r1 | yes |

**The reading rules, applied:**
- `lp_probe` and `ct2a2_probe` bring most of the lost wins, and `probe` does not. So what carries the wins is
  practising a fixed number of routines every round, ranked by cost or by looks, from a uniformly probed pool. The
  threshold is what loses them.
- `lp` brings 1 of hardest first's 12 where `lp_probe` brings 11: hardest first needs the uniform probe.
- `k4` against `lp`: both bring almost nothing without the uniform probe, so this comparison says nothing about count
  against ranking. Addendum 2's `kmax_probe` asked it with the probe (§7.3).

#### 7.3 Addendum 2 (one rule for all the wins, ab2)

| arm | predicted | came | right? |
|---|---|---|---|
| `ct2a2_probe`, the 9 | at least 6 of anchored pairs' 8; not 10·x² + 1 (medium-high) | 6: x² mod N r0, 2x + 1 r0, x² + 3 r1, x³ r1, (x + 1)² r1, 3·x² r1. Not come when stopped: x⁴ and 10·x² after 4 rounds, 10·x² + 1 after 3 | yes, at its bound |
| `ct2a2_probe_noties` | the same as `ct2a2_probe`, but for at most 2 (medium) | the same 13 wins at the same rounds; x⁴ and 10·x² stopped after 4 rounds, as with the ties | yes |
| `kmax_probe` | 6 to 11 of 16 (low) | 4: 2x + 1 r0, (x + 1)² r1, 2^x r1, binary x mod N r3. Fibonacci no; 11 stopped after 2–4 rounds | no |

**The reading rules, applied:**
- `ct2a2_probe` brings 13 of anchored pairs' 15, not all. The two it lost or left undecided are x⁴ and 10·x², the two
  where the third rung changed its anchors (§6.2). So one old rule carries every earlier win but three: 10·x² + 1, x⁴
  and 10·x². The last two it loses to the third rung that replaced the subgoal's deeper probe, not to its own ranking.
- `ct2a2_probe_noties` brings the same: the subgoal's tie-break can go.
- `kmax_probe` brings 4 against 13: here, one score did not stand for both choosing rules.

### 8. Dead ends, and how hard each was tried

- **The one-score allocator with a gain threshold (PREREG.md).**
  - Tried once on each of 21 formulas (cs1), with no change after its first results. It kept 1 of 16 earlier wins.
  - The cause is the myopic policy's: a 30k look underestimates a routine's worth, and the threshold makes that a veto
    (§5.2, §2.0).
  - Not tried:
    - deeper first looks. The 1M rung was reached only after a positive 30k gain, and few looks had one.
    - pair looks inside the allocator;
    - a library cost other than the merge's weight;
    - R in states rather than nominal budgets;
    - a stop rule for a round that practises nothing.
- **The threshold with the uniform probe (`probe`).**
  - Tried once on each of the 15 lost wins. 4 came.
  - With more candidates solved, the threshold still practised nothing in 29 of its 54 rounds (x³: nothing in all 5).
    Where it did practise, the count varied from 1 to 23 a round.
- **A fixed count without the uniform probe** (`lp`, `k4`, `ct2a2`).
  - Tried once on each lost win (anchored pairs on its 7). Each brought 1 or 2.
  - The costly candidates the old rules practised are solved only by a 1M probe. The allocator gave that probe to about
    a quarter of the candidates, by value.
- **Four picks by max(measured gain, prior), with the uniform probe (`kmax_probe`, addendum 2).**
  - Tried once on each of the 16 wins. 4 came, and 11 were stopped after 2–4 rounds.
  - Its prior includes the subgoal score's ~150 bits for a table candidate, so it practised those first.
  - Not tried: the same score without the table score's bits, or with pair looks.
- **The third rung by value, in place of the subgoal's separately budgeted deeper probe** (every arm, by design).
  - It did the subgoal's work on x³ and x² + 3, but over-solved on x⁴ and 10·x² (§6.2).
  - Not tried: the rebuilt rules with the old subgoal-only deeper probe.

### 9. What is left

- **A non-myopic allocator.**
  - The one-score idea failed here through its value estimate, not through its form.
  - A value that a null look cannot veto was tried only with the subgoal score in the prior (`kmax_probe`, §6).
  - Untried:
    - a look's gain taken as a lower bound and its depth's chance of missing taken into account;
    - the value of a routine in its best pair (anchored pairs do this by rule);
    - a look after a round of practice, which is cheap because a learnable routine needs no practice search.
- **The third rung's allocation.**
  - Taken by value, the 10M probe went to the subgoal score's table candidates. min(N∸x, x) is the most common.
  - Solved, such a candidate becomes one of the costliest. That changed hardest first's picks (10·x²: 1 of the old 4)
    and anchored pairs' anchors (x⁴: 0 of 4; x² + 3: 2 of 4).
  - Whether the deeper probe belongs to the subgoal alone, or to a value that knows the table score's optimism, is
    open.
- **10·x² + 1** came only with hardest first and the rest's tie-break (bar3/scale/picks). That is the one patch with a
  win that anchored pairs does not cover.
- **The composition harness** was not tried. The coordinator's condition, that the counting machine go well, did not
  hold for the allocator.
- **The bug fixes:**
  - The row check was on in every arm.
  - Bar 3's machine starts clean.
  - Its search lists every spelling of a goal state (one per instruction class, up to 200 programs a level), so
    stage 3's one-spelling blind spot, which SPELL fixes, is not in it.

### 10. Cost (`cost.py`; `logs/cost.txt`, `logs/watch_*.txt`)

- **Billed to 21:00 UTC: $48.48.**
  - By part: cs1 $3.47, ab1 $25.12, ab2 $19.89.
  - By hour: 18:00 UTC $2.38, 19:00 $18.79, 20:00 $27.32.
  - The four minutes of the 21:00 UTC hour before the stop are not billed yet: about $1 more, at the containers'
    rate.
- **About $49.5 in all:** under the $50 cap, about $9.5 over the soft $40.
- **Why it went over the soft limit:**
  - The billing report lags about an hour, and the watcher's estimate takes the hour under way at the last complete
    hour's rate.
  - Addendum 2's 16 groups started at 12:44 PDT, beside ab1's 15. So at 13:45 PDT the estimate for the whole study
    was about $32, and my estimate for addendum 2, $12, from cs1's $0.05 per billion states, was low.
    - The containers bill by the hour (about $1.15 a container-hour), not by states.
    - The arms with the uniform probe ran longer.
  - When the 20:00 UTC hour posted, the watcher's estimate jumped to $47.83, past its $44 total for the ablation's
    apps. It stopped every app at 14:04 PDT, as set.
  - Lesson: with an hour's billing lag, a total for many containers started mid-hour has to be set about one hour of
    their rate below the cap.
- **The stops:**
  - 13:46 PDT, by hand, under the addenda's money rule (low-priority runs first): 4 ab1 groups, 9 runs.
  - 14:04 PDT, by the watcher: the 13 groups still running, 21 runs.
  - Every app of this study is stopped.

### 11. Files and how to rerun

**Design and records:**
- `PREREG.md`: the allocator's pre-registration (read-only).
- `PREREG_ADDENDUM1.md`: the ablation's pre-registration (read-only).
- `PREREG_ADDENDUM2.md`: one rule for all the wins (read-only).
- `alloc.py`: the allocator, with the ablation's flags (`alloc_probe_all`, `alloc_pick` = `lp` / `k` / `kmax`).
- `make_learners.py`: writes `learn_newton.py` and `learn_digits.py` from bar3/scale/reach's copies (4 asserted
  replacements).
- `configs.py`: the three allocator configs, and every arm's.
- `modal_consolidate.py`: the runs on Modal (app and Volume `bar3-consolidate`).
- `records/`:
  - `*_cs1.jsonl`: the allocator's runs, a line per round, with each round's `alloc` record;
  - `*_ab1.jsonl`: the ablation's runs;
  - `*_ab2.jsonl`: addendum 2's runs;
  - `calls_*.json`: the Modal call ids.
- `logs/`:
  - the launches;
  - the spend watchers;
  - the tiny tests and their store;
  - the replay checks;
  - cs1's mechanism counts (`cs1_mechanisms.txt`, `cs1_oldpicks.txt`, `compare_cs1.txt`);
  - the final table (`table_final.txt`), and ab1's table when addendum 2 was frozen (`ab1_at_addendum2.txt`);
  - the cost (`cost.txt`);
  - the addenda's hashes.

**Tests and reports:**
- `tiny.py`: the local tiny test, with flags.
- `check.py`: the replay check with the allocator off.
- `compare.py`: each run against the earlier records, per round.
- `table.py`: the ablation's table.
- `cost.py`: the study's cost from the billing report.

**To rerun** (from `cpu_first`, with numba's venv for the local parts):
- **The local tests:**
  - `bar3/scale/rowcheck/memcap.sh 10 <python> -m bar3.scale.consolidate.tiny h1 [FLAG=JSON ...]`
  - `bar3/scale/rowcheck/memcap.sh 10 <python> -m bar3.scale.consolidate.check off STORE ... -- RECORD ...`
- **A run on Modal:**
  1. `~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750`
  2. `~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/consolidate/modal_consolidate.py --fresh CONFIGS
     --targets T,... --tag TAG --store STORE.jsonl`
  3. The joined store `v1/STORE.jsonl` is on the Volume (87 MB, not in the repo). Every later run's store,
     `v1/store_<first run>.jsonl`, is there too.
- **The records:** `~/github/sotaku/venv/bin/python -m modal run bar3/scale/consolidate/modal_consolidate.py --fetch
  [--only TAG]`
  - They are committed gzipped, as `records/*.jsonl.gz` (9 MB instead of 96 MB), and `compare.read` reads either
    form. A fetch writes plain `.jsonl`.
- **The tables:**
  - `~/github/sotaku/venv/bin/python -m bar3.scale.consolidate.compare TAG --rounds`
  - `~/github/sotaku/venv/bin/python -m bar3.scale.consolidate.table`
