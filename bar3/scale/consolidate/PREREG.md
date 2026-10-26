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
6. **Practice:** every learnable candidate whose measured gain exceeds its library cost (its program's tokens × tok × 0.5,
   the merge's library weight). No fixed count, and nothing practised when none passes. Then the merge and the target
   search, as before.

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
