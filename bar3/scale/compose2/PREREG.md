# Composition round 2: pre-registration

Written 2026-10-26 07:41 PDT, before any run of this study. COMPOSE2.md §1 repeats this text unchanged.

## The question

Round 1 (bar3/scale/compose/COMPOSE.md; 9e24be4) found that on stage 3's digit machine only the law book gets the
round. Bar 3's sources posed no piece of it, alone or in place of the book's last law. Round 2 ports the rest of bar
3's pieces (the subgoal score, the row-based duplicate check, the tie-break) and fixes what round 1 found. With all of
it, does bar 3's full set get the round, alone or with book_min less its reduction?

## Theory, in brief

Round 1's four blockers, and what each piece can do to them:
1. **The value source posed dirty values**, which are not functions of x and N. The fix is round 1's `clean.py`, on.
2. **The multiply's route (the rows where x² < N) was never posed.** The source measures a predicate's share on the
   target's 8-probe draw, and there x'² < N' held on none. The fix is to measure it on many rows.
3. **x² was never a near miss.** Ranked by fit it fits 3 of 16 probes, against 6–10 for the near misses kept.
   - Bar 3's subgoal score sees what fit does not: x² mod N = mod(g, N) for g = x², one law away (SUBGOAL.md §1).
   - But its rule turns the law view off for a target that is itself a law of X and N of depth ≤ 2, and x² mod N =
     mod(mul(X, X), N) is one. So the rule is split by depth here. Depth 1 stays on, since x² mod N is no single law
     of X and N. Depth 2 is off.
   - Applied to the near-miss search's states as well as to posed problems, the law view poses x². That needs the
     multiply, which only the book supplies.
4. **Handed the rest of the way from x², the search accepted nothing**, though a cost-5 reduction is right and accepted.
   - `bfs_roots` keeps one machine state per class of programs that agree on the probes, across levels.
   - It tests one spelling of each goal state (`programs(..., limit=1)`), at most 50,000 goal states a level.
   - That is enumeration modulo observational equivalence on the probes (Udupa et al., PLDI 2013; Albarghouthi,
     Gulwani and Kincaid, CAV 2013). It is lossless when the probes decide acceptance, and lossy when acceptance uses
     more rows: a class's tested member can fail while another passes.
   - Counterexample-guided synthesis (Solar-Lezama et al., ASPLOS 2006) would add the failing rows to the probes.
     The cheaper fix is to test more members.

The other two pieces touch no blocker directly:
- **The row check** (ROWCHECK.md) decides aliases and duplicates on a problem's own rows instead of by signature on
  random machines.
- **The tie-break** (PICKS.md's tiescore) scores candidates against the rest of the way from a practised subgoal. For
  the rest from x², the answer is mod(X, N), itself a law of the rest's inputs, so only the table view acts.

So:
- **Alone** (no book), nothing poses the comparison, the link, the addition or the row. The multiply (9 tokens over the
  primitives) stays out of reach even with x² < N posed.
- **With book_min less its reduction**, the route is:
  1. the multiply, from the book;
  2. x², posed by the law view;
  3. the rest of the way from x², posed by the rest source;
  4. that rest learned, if the search tests the right spelling;
  5. the round, 3 tokens over the routines: x²'s routine, `STA.X`, the rest's routine.

## The test of the guess (first, on a B200; `spell_test.py`)

Round 1's route setup: A2c's final library, the rest of the way from x² on the arms' draws, the deep probe's most
favourable vocabulary (the primitives with T.red.pair.AN and T.lt.pair.AN), 5.8M states. Measured:
- the hand-written reduction `CLA P[BRG.X B[T.red.pair.AN]]`'s final machine state on the probes;
- each level's goal states and the spelling tested for each.

Which one holds:
- **(a)** the reduction's state is first reached at a lower cost by another program, and only that spelling is tested;
- **(b)** it is reached at cost 5, and another spelling of the same state is tested;
- **(c)** it is not reached at all;
- **(d)** it is reached, past the 50,000 goal states a level that get a spelling.

The fix, flag SPELL, is designed after the test and frozen in an addendum before any arm:
- for (a) or (b), more spellings per goal state when the first fails, (a) also across levels;
- for (d), a larger cap;
- for (c), none: the guess is wrong.

## The flags (`make_copies.py`: spcore_c2.py, oracle_c2.py, core_c2.py, climb_c2.py, clean_c2.py; `subgoal_c2.py`)

Each behind its own env flag in asserted copies. With every flag off (CLEAN_VALUES too), the copy is round 1's driver.
- **SUBGOAL: bar 3's subgoal score** (SUBGOAL.md §1, score.py).
  - Scored: every candidate posed on the target's own root with its answer in A, i.e. the near misses' values and
    the projections low and high. Not restrictions (part of the rows) or rests (other inputs).
  - Its rows: the target's 16 probes and the acceptor's 60 rows at each verify width.
  - Its views: score.py's table view and law view (the book's operations, leaves g, X, N, every value below the
    register's capacity), with the off rule split by depth as above.
  - The subgoal is the unique best positive key. It gets one deeper probe (10M states) when the probes leave it
    unsolved, and it wins the ties pool order breaks: under lp, among the learnable at its cost; under lpca, among
    the concepts at its hardness.
- **LAWNEAR: the law view ranks near misses too.** The look at each level's pose also takes its held states whose A,
  X, Y or S passes the law view (depth 1) on the target's probes: up to 4, cheapest first, beside the 12 best by fit.
  Their values are posed as every near miss's are.
- **ROWCHECK** (ROWCHECK.md): in the climb's classify,
  - a learnable candidate with a token's signature is an alias only when that token passes the candidate's own
    acceptor;
  - a duplicate only when the earlier learnable's program with that signature passes it;
  - otherwise it is learnable, and joins its signature's list.
- **TIEBREAK** (PICKS.md's tiescore). Once the subgoal is practised and the rest of the way from its value is posed,
  every learnable candidate bound to X and N is run on that rest's probes and scored against the rest's answers by
  the subgoal score's rule. The one it singles out wins the ties.
- **CLEAN_VALUES:** round 1's `clean.py`. On by default this round.
- **RESTRICT_ROWS:** the restriction source measures each predicate's share, and keeps one predicate per pattern, on
  the climb's 4,096 rows' top parts at the target's last width, not on its 8-probe draw. The 10–80% rule stays, and
  each restriction's probes are drawn where it holds, as before.
  - Chosen over choosing the width where x² < N holds often: it keeps the source's rule and the climb's widths, and
    removes only the small sample's noise.
  - Plainly: this chooses which rows decide what is posed. That is allowed offline.
- **SPELL:** after the test.

## Runs

Each on one B200, as round 1: TARGET small, ROUND_TOO, NP 8, deep probes over whole concepts, LEVELS 10, SECONDS
2,400, every flag on (SPELL as the addendum fixes it):
- **B1:** bar 3's full set alone, lpca. **B1lp:** the same under lp.
- **B2:** bar 3's full set with book_min less x + o N (round 1's A2), lpca. **B2lp:** the same under lp.
- **If an arm gets the round:** that arm with each flag off in turn.
- **Replays first:** cp_none, cp_book_min and cp_bar3_clean with every new flag off must give round 1's records,
  apart from times (`same.py`). Then round 1's book_min and no-practice records are this round's controls.

A round is checked on both of H1's ladders to T = 64. Money: soft $30, cap $40, `spend.py 4750` before each launch,
one spend watcher, local runs under a memory cap.

## Predictions

| what | prediction | confidence |
|---|---|---|
| the guess's test | the reduction's state is a goal state tested in another spelling, (a) or (b) | medium |
| | (c) or (d) | low |
| the replays | round 1's records, apart from times | high |
| B1 | x² < N posed from level 0 | high |
| | not learnable: it needs the multiply | high |
| | none of the six pieces; no small target, no round | high |
| B1lp | the same | high |
| B2 | the five pieces at levels 0–4 | medium-high |
| | after the multiply, x² among the near misses | high, given the multiply |
| | x² the subgoal, practised | medium-high |
| | the rest of the way from x² posed | certain, given the line above |
| | that rest learned, a reduction accepted at width 3 | medium with SPELL |
| | the round | medium-low |
| | if the round comes, right on every rung of both ladders | high |
| B2lp | the same as B2, less likely: lp's picks do not follow concepts | low-medium for the round |
| the tie-break | singles out nothing, in every arm | medium-high |
| the row check | changes some statuses, and nothing on the route | medium |
| one flag off, if a round comes | LAWNEAR off or SPELL off: no round | medium-high |
| | the others off: the round still comes | medium |
