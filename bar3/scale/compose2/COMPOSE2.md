# Composition round 2: bar 3's full set, fixed, in place of the law book's last law

<!-- PLAIN -->
## In plain words

**The question.** Round 1 (bar3/scale/compose/COMPOSE.md) found that on stage 3's digit machine only the law book gets
the round, and named four blockers. Round 2:
- ports the rest of bar 3's pieces: the subgoal score, the row-based duplicate check, the tie-break;
- fixes the blockers, each behind its own flag;
- tests round 1's guess about the search directly.

Does bar 3's full set get the round, alone or in place of the book's last law?

**The guess held, and its fix worked.** Handed the rest of the way from x², the search kept the reduction's machine
state but tested another spelling of it, a look-alike that fails at width 3. Testing up to 16 spellings a state
(SPELL) accepts the reduction on the same setup.

**What came: the round, with the book's last law replaced, once one more fix was added.**

| arm | what | the round |
|---|---|---|
| B1, B1lp | bar 3's full set alone (lpca, lp) | no: nothing learnable from level 2 (B1) and to the cap (B1lp) |
| B2, B2lp | bar 3's full set with book_min less its reduction | no: both to the 2,400 s deadline |
| **B2rd** | B2 with REST_DEEP | **yes, level 5 (625 s)** |
| B2lp_rd | B2lp with REST_DEEP | no: to the deadline. lp's link in A against N reads X, so no 5-token reduction exists |
| controls (round 1, reproduced) | book_min; no practice | yes, level 5; no |

**B2rd's round** is `T.val.126.tgt.XN STA.X T.target.rest.126.XN`:
- x² (`LDA.X STA.Y T.mul.x,o.XY`), posed by the law view and singled out by bar 3's subgoal score;
- then the reduction `LDA.N P[BRG.X B[T.red.pair.AN]]`, learned from the rest of the way from x².

It is also the first small target in either round, found at cost 3. It is right on all 400 answer rows at full size,
and on every rung of both of H1's ladders (T = 1 to 64, seen and unseen moduli, 768/768 each).

**How the route goes, and what each fix did:**
1. **The multiply** comes from the book (levels 0–4: the comparison, the link, the addition, the row, the multiply).
2. **x² is posed by the law view.** It fits 3 of the 16 probes against the near misses' 5–6, so fit alone never kept
   it. But it is one law, mod(g, N), from the target. That needed splitting bar 3's own off rule by depth, since
   x² mod N = mod(mul(X, X), N) turns the whole law view off otherwise (LAWNEAR).
3. **x² is the subgoal:** the only candidate with a law to the target (key (1, 0); SUBGOAL).
4. **The rest of the way from x² is posed** by bar 3's rest source.
5. **The reduction is learned** only with two fixes:
   - REST_DEEP: the subgoal's deeper probe aimed at its rest. Without it, the probes never reached cost 5, and the
     deep probe's focus missed the link the reduction needs (B2);
   - SPELL. Without it, the deeper probe keeps the reduction's state and tests a look-alike spelling.
6. **The round:** 3 tokens over the two routines.

**One fix off at a time** (on B2rd, to level 6):

| off | the round |
|---|---|
| REST_DEEP (= B2) | no |
| SPELL | no |
| LAWNEAR | no |
| SUBGOAL (REST_DEEP aims at the subgoal) | no |
| CLEAN_VALUES | yes, level 5 |
| ROWCHECK | yes, level 5 |
| RESTRICT_ROWS | yes, level 5 |
| TIEBREAK | the same run: it never acted |

**What did nothing:**
- **The multiply's route** (x² < N posed from the rows where it holds, RESTRICT_ROWS) is posed now. But alone, with no
  row to build it from, the multiply is 9 tokens over the primitives, and B1 learned nothing from it. With the book it
  gave one of the round's two spellings (`T.target.rst5.XN`).
- **The tie-break never acted.** The value source names each level's values afresh, so the practised subgoal's rest
  was never posed again under its name. In B2rd the round came in the level the subgoal was first practised. A port
  limit, reported, not fixed.
- **The row check** changed which candidates count as duplicates (5 in B2, 38 in B2lp), and nothing on the route.
- **Clean values** were not needed. Without them the round came at level 5 too, more slowly: LAWNEAR adds x² beside
  the near misses, whatever they are.

**What it means.**
- **On stage 3's machine, bar 3's own sources replace the book's last law.** Given the multiply, the reduction comes
  from the target's rows alone: the near miss holding x², the rest of the way from it, and the law test that sees
  x² mod N = mod(x², N).
- **The five pieces below the reduction still come from the book:** the comparison, the link, the addition, the row
  and the multiply. Bar 3's set alone gets nothing.
- **Four flags carry the result,** each needed (one off, no round):
  - the law view ranking near misses, which poses x² (LAWNEAR);
  - bar 3's subgoal score, which singles x² out and so aims the deeper probe (SUBGOAL);
  - the deeper probe aimed at the subgoal's rest (REST_DEEP);
  - the search testing more than one spelling (SPELL).

  SPELL's need is a property of stage 3's search, not of bar 3: a search modulo observational equivalence on the
  probes, with acceptance on more rows.

**What is left** (§8):
- The comparison, the link, the addition, the row and the multiply still need the book.
- The tie-break needs a name for the practised subgoal's value that lasts past its level.
- SPELL lives in this study's copy only; the committed search keeps the blind spot.
- Under lp the route needs a clean link, and lp practised one that reads X.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md (frozen 07:41 PDT, sha256 `d98c2465885134f8588f01e44a0758c89e7a483479378f98579811705fa59a84`),
PREREG_ADDENDUM.md (07:53, `2ce228b223313a34c116106cd499ff439ad29eedcda6d1f43a8fcb4e155209c6`, SPELL) and
PREREG_ADDENDUM2.md (08:43, `ccf272a4aafa7456cae496686ad5cc64188ce2c44666cded1e135a32238b8629`, REST_DEEP), each
frozen (read-only) before the runs it names, unchanged:

<!-- PREREG -->
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

# Composition round 2: the test of the guess, and SPELL, fixed before any arm

Written 2026-10-26 07:53 PDT, after the test of the guess and before any arm of this study. COMPOSE2.md §1 repeats
this text unchanged.

## What the test found (`spell_test.py` on a B200, 07:45 PDT; `records/spell_test.log`)

On round 1's route setup, the committed search over the primitives with T.red.pair.AN and T.lt.pair.AN, at 5.8M
states:
- The two hand-written reductions, `CLA P[BRG.X B[T.red.pair.AN]]` and `LDA.N P[BRG.X B[T.red.pair.AN]]`, end in the
  same machine state on the probes. Both reach the goal on the probes, and both are accepted at width 3.
- **That state is held at level 5**, one of its 22 goal states (level 4 had 1). It has 6 spellings. The one tested
  is the first, `P[BRG.X] B[P[T.red.pair.AN]]`. It fails acceptance. Both reductions are among the other five, and
  both pass.
- The acceptor was handed 23 programs in all, one a goal state. Neither reduction was among them.

So the answer is **(b)**:
- not (a): the state is first reached at cost 5;
- not (c): it is reached;
- not (d): 22 goal states, far under the 50,000 that get a spelling.

The guess holds.

## The fix (flag SPELL, in `spcore_c2.py` only)

With SPELL=k, `bfs_roots` takes up to k spellings of each goal state (`programs(..., limit=k)`) instead of one.
- Every spelling of every goal state is sorted as before, by top-level items and then primitives.
- They are tested in batches of 256 until a batch has one that passes.
- At most 50,000 programs a level.

The arms run SPELL=16. With SPELL=0 (unset) the copy tests one spelling a state, as the committed search does. The
round's own search (libcore.bfs, whose probes are the answers) is untouched.

## Predictions

| what | prediction | confidence |
|---|---|---|
| the test's setup on the copy with SPELL=16 (`spell_fix`) | a reduction accepted at level 5 | certain, given the test |
| the arms | as PREREG.md | |

# Composition round 2: REST_DEEP, two more arms, pre-registered before their launch

Written 2026-10-26 08:43 PDT. PREREG.md's arms were running: B1 and B1lp had ended (no round), B2 was at level 7 and
B2lp at level 6. These arms were not launched. COMPOSE2.md §1 repeats this text unchanged.

## Why

B2 (bar 3's full set with book_min less its reduction) took the route as far as its fourth link:
1. the multiply, from the book, at level 4;
2. at level 5, x² posed by LAWNEAR (`LDA.X STA.Y T.mul.x,o.XY`, right on 3 of 16 probes; the 12 by fit, on 5–6),
   singled out by the subgoal score (key (1, 0), mod(g,N)), and practised as `T.val.126.tgt.XN`;
3. the rest of the way from it posed, as `rest.126@XN`.

But it never solved that rest:
- The probe (1M states over the composition vocabulary) does not reach cost 5 with a library like this one. On
  round 1's A2c library, the rest's level 4 alone held 5.6M states.
- The deep probe's focus is the closest attempt's routines. Here those were none, or `T.lt.pair.AX`, never the link
  `T.red.pair.AN` the reduction calls.
- Later levels practised x² less multiples of N (`T.target.rst5.XN B[B[T.red.pair.AN]]`). Those are right on all 16
  practice probes and wrong at width 3.

Two checks on round 1's route setup (A2c's library; on a B200):
- `spell_full`: the composition vocabulary at 10M states with SPELL=16 accepts `CLA P[BRG.X B[T.red.pair.AN]]` at cost
  5, in 71 s.
- `rest_deep_test`: the hook below, end to end. The rest from x², posed with the subgoal x² and left open, gets
  `LDA.N P[BRG.X B[T.red.pair.AN]]` and `CLA P[BRG.X B[T.red.pair.AN]]`, both accepted.

## The change (`hooks.rest_deep`, flag REST_DEEP)

SUBGOAL's deeper probe goes to the subgoal when the probes leave it unsolved. A value subgoal is solved at once (its
own prefix computes it), so with REST_DEEP the deeper probe goes one step on instead:
- to the rest of the way from the subgoal's value, posed with it this level;
- when the probes have left that rest unsolved;
- at the same 10M states, over the composition vocabulary (every routine; the focus off).

With REST_DEEP off the hook returns at once, and the copy is the one the earlier arms ran. The diff is the flag's
lines (`logs/core_c2_main_arms.py.txt`, `logs/spcore_c2_main_arms.py.txt`).

## The arms

- **B2rd** (`c2_b2_rd`): B2 with REST_DEEP.
- **B2lp_rd** (`c2_b2_lp_rd`): B2lp with REST_DEEP.

Everything else as B2 and B2lp. If either gets the round:
- that arm with each flag off in turn, to the level after the round's (LEVELS);
- REST_DEEP, LAWNEAR, SUBGOAL, SPELL and CLEAN_VALUES first, as the flags on the route;
- ROWCHECK, TIEBREAK and RESTRICT_ROWS if the money allows.

## Predictions

| arm | prediction | confidence |
|---|---|---|
| B2rd | levels 0–5 as B2's, up to REST_DEEP's probe at level 5 | high |
| | REST_DEEP finds a reduction of the rest from x², accepted at width 3 | medium-high |
| | it is practised | high, given the line above |
| | the round comes, by level 6 | medium |
| | if it comes, right on every rung of both ladders | high |
| B2lp_rd | the same route, later (lp reached the multiply later) | low-medium for the round |
| one flag off, if the round comes | REST_DEEP, LAWNEAR, SUBGOAL or SPELL off: no round | high |
| | CLEAN_VALUES, ROWCHECK, TIEBREAK or RESTRICT_ROWS off: the round still comes | medium |
<!-- /PREREG -->

### 2. Theory and design

#### 2.1 Theory

**Round 1's four blockers** (COMPOSE.md §5):
1. the value source posed values that read the dirty start;
2. the rows where x² < N, the multiply's specification, were never posed;
3. x², right on 3 of 16 probes, was never among the near misses kept by fit (6–10 of 16);
4. handed the rest of the way from x², the search accepted nothing, although a cost-5 reduction is right and accepted.

**Bar 3's three pieces, and what each can reach:**
- **The subgoal score** (SUBGOAL.md §1) reads a candidate's answers on the target's rows and asks whether the target
  factors through them:
  - by a table, a functional dependency (g, N) → t;
  - or by one law of the book, t = op(g, b), or two.

  It sees a piece before it is solved, so it can rank x² where fit cannot: x² mod N = mod(x², N), one law away. Two
  obstacles stand in the way:
  - **its own rule** turns the whole law view off for a target that is itself a law of X and N of depth ≤ 2, because
    then any copy of X passes. x² mod N = mod(mul(X, X), N) is such a target. Split by depth, the rule is exact for
    what it guards against: a copy of X passes at depth 2, never at depth 1, since no single law of X and N is
    x² mod N. Depth 1 stays on;
  - **it scores what is posed**, and x² is not posed (blocker 3). So the law view also has to rank the near-miss
    search's states, not only posed problems (LAWNEAR).
- **The row check** (ROWCHECK.md) is observational equivalence on a problem's own examples (Albarghouthi, Gulwani and
  Kincaid, CAV 2013). It replaces equivalence on 24 random machines, and changes which problems count as aliases and
  duplicates. No blocker is about that.
- **The tie-break** (PICKS.md's tiescore) scores the pool against the rest of the way from a practised subgoal. For the
  rest from x², the rest's answer is mod(X, N), a law of its own inputs, so the law view is off by the same rule, and
  only tables act.

**Blocker 4 is a property of the search, not of bar 3.**
- `bfs_roots` keeps one machine state for every class of programs that agree on the probes, and tests one spelling a
  goal state. That is enumeration modulo observational equivalence (Udupa et al., PLDI 2013; Albarghouthi et al.
  2013).
- It is lossless when the probes decide acceptance, and lossy when acceptance asks more, here 60 fresh rows a width
  and width 3: a class's first spelling can fail where another passes.
- Two remedies are known:
  - refine the classes with the failing rows, as counterexample-guided synthesis does (Solar-Lezama et al., ASPLOS
    2006);
  - or test more members of a class, the one taken here.

**Expected:**
- **Alone,** no source poses the comparison, the link, the addition or the row. The multiply is 9 tokens over the
  primitives, so even its posed specification cannot be learned.
- **With book_min less its reduction,** the route opens if every link holds:
  1. the multiply, from the book;
  2. x², posed by LAWNEAR;
  3. x², the subgoal: the only candidate one law from the target;
  4. the rest of the way from x², posed by the rest source;
  5. the reduction, found by the probes and accepted with SPELL;
  6. the round: x²'s routine, `STA.X`, the reduction's routine.

#### 2.2 The copies (`make_copies.py`)

Five copies of committed files, written with asserted replacements (each matched exactly once, 26 in all, marked
"compose2"). The changes live in `hooks.py` and `subgoal_c2.py`; each hook returns at once when its flag is off.

| copy | from | replacements |
|---|---|---|
| `spcore_c2.py` | integration/self_practice/spcore.py | 12: SPELL in bfs_roots; the tokens by signature; ROWCHECK in classify; the ties in hardest and concepts; SUBGOAL's scores after each pose, and its deeper probe before classify |
| `oracle_c2.py` | integration/self_practice/oracle.py | 4: the copy's search; RESTRICT_ROWS; LAWNEAR |
| `core_c2.py` | integration/self_practice/core.py | 8: round 1's (";" lists, ROUND_TOO, CLEAN_VALUES) and the flags from the env, recorded in the config only when on |
| `climb_c2.py`, `clean_c2.py` | round 1's climb.py and clean.py | 1 each: on the copies |

#### 2.3 The flags

- **SUBGOAL** (`hooks.subgoal_scores`, `subgoal_deep`, `tie_rank`; `subgoal_c2.py`):
  - *the score:* bar 3's score.py, copied (`check_subgoal.py` checks the copy against it). It is applied after
    each pose to the candidates posed on the target's root with answers in A: the near misses' values and the
    projections low and high;
  - *its rows:* the target's 16 probes and the acceptor's 60 rows at each verify width, each (x, N) once;
  - *the law view's off rule, by depth* (§2.1);
  - *what the subgoal does:* it is the unique best positive key (a law first, then the rows a table saves). If the
    probes leave it unsolved it gets one deeper probe (10M states), and it wins the ties that pool order breaks.
- **LAWNEAR** (`hooks.law_near`): the look's held states (levels 1 up to the look's budget) whose A, X, Y or S passes
  the depth-1 law view on the target's probes join the near misses. At most 4, cheapest first, distinct values.
- **ROWCHECK** (`hooks.row_check`): an alias only when a token with the candidate's signature passes the candidate's
  own acceptor; a duplicate only when an earlier learnable's program with the signature does.
- **TIEBREAK** (`hooks.tie_candidate`): once the subgoal is practised and the rest of the way from its value is posed,
  every learnable candidate bound to X and N is run on that rest's probes and scored against its answers. The one
  singled out wins ties after the subgoal.
- **CLEAN_VALUES**: round 1's clean.py, on in every arm.
- **RESTRICT_ROWS** (`hooks.share_rows`): each predicate's share and pattern are measured on the climb's 4,096 rows'
  top parts at the target's last width.
- **SPELL=16** (in bfs_roots; PREREG_ADDENDUM.md): up to 16 spellings a goal state, at most 50,000 programs a level.
- **REST_DEEP** (`hooks.rest_deep`; PREREG_ADDENDUM2.md): when the subgoal is solved (a value is, at once), SUBGOAL's
  deeper probe goes to the rest of the way from its value posed with it, if the probes left that open: 10M states over
  the composition vocabulary, the focus off.

### 3. The tests before the arms

| test | where | what it showed |
|---|---|---|
| `check_subgoal.py` | local | subgoal_c2's operations, table views and depth-2 laws equal score.py's on 1,800 random cases; on x² mod N the law view is on at depth 1 only, and passes x² (mod(g,N)) and not x |
| the copies with every flag off (`logs/off_*`) | local, tiny budgets | round 1's tiny records for book_min, bar 3's sources and clean values, apart from times (`same.py`) |
| every flag on (`logs/on_b1*`, `logs/on_b2*`) | local, tiny budgets | the hooks run; the subgoal score reads 157 rows, its law view at depth 1; no multiply at these budgets |
| `spell_test` | B200 | the guess, (b): the reduction's state is held at level 5 and tested in a look-alike spelling (PREREG_ADDENDUM.md) |
| `spell_fix` | B200 | with SPELL=16 the same search accepts the reduction at cost 5 (98 programs tested) |
| `lawnear_test` | B200 | on A2c's library, LAWNEAR adds x² (fit 3) to the 12 near misses (fit 6), its value is posed, the subgoal score singles it out (key (1, 0), mod(g,N)) and its rest is posed |
| `restrictions_c2.py` | local | with RESTRICT_ROWS six restrictions are posed, x² < N among them (15% of the rows); its probes all have x'² < N' |
| `spell_full` | B200 | the composition vocabulary at 10M states with SPELL=16 accepts the reduction at cost 5 (71 s) |
| `rest_deep_test` | B200 | REST_DEEP end to end: the rest from x² gets `LDA.N P[BRG.X B[T.red.pair.AN]]` at cost 5, accepted |

### 4. The runs (`modal_compose2.py`; `records/*.json` and `.log`)

Each on one B200, from round 2's copies, round 1's settings: TARGET small, ROUND_TOO, RULE lpca unless named, NP 8,
deep probes over whole concepts, LEVELS 10, SECONDS 2,400. "All" is CLEAN_VALUES, SUBGOAL, LAWNEAR, ROWCHECK,
TIEBREAK, RESTRICT_ROWS and SPELL=16.

| tag | arm | SOURCES | LAWS / PATTERNS | flags |
|---|---|---|---|---|
| `replay_cp_none`, `replay_cp_book_min`, `replay_cp_bar3_clean` | round 1's controls | as round 1 | as round 1 | none new |
| `c2_b1`, `c2_b1_lp` | B1, B1lp | restrict,project,value,rest | | all |
| `c2_b2`, `c2_b2_lp` | B2, B2lp | law,restrict,project,value,rest | lt;red;add;mul;acc.mul / pair;x,o;o,x,d | all |
| `c2_b2_rd`, `c2_b2_lp_rd` | B2rd, B2lp_rd | as B2 | as B2 | all and REST_DEEP |
| `c2_b2_rd_no_<flag>` | one flag off | as B2 | as B2 | all and REST_DEEP but one; LEVELS 7 |

Launches (PDT), each after `spend.py 4750` (LAUNCH OK; $3,605.94 to $3,624.49 billed this month), one spend watcher
over every app (stop each at $36, all at $36; `logs/watch_c2_*.txt`):
- the tests from 07:44;
- the replays and the four main arms at 08:13;
- the REST_DEEP arms at 08:43;
- four flags off at 08:57, two more at 09:10.

The replays reproduce round 1's records exactly, apart from times, on the B200: cp_none, cp_bar3_clean, and
cp_book_min (its round at 582 s against 697 s). So round 1's book_min and no-practice records are this round's
controls.

### 5. Outcomes (`logs/verify_c2.txt`)

**B1 and B1lp, bar 3's full set alone.**
- **Posed:** six restrictions a level, x² < N among them; the projections; clean values (one learnable: 10 x).
- **Never learnable:** any restriction, projection or rest.
- **No law view hit:** without a multiply no state holds x², so neither LAWNEAR nor the subgoal score found anything.
- **How they ended:**
  - B1: nothing learnable at level 2 (72 s);
  - B1lp: nothing new to the 10-level cap (270 s).

  No small target, no round.

**B2, bar 3's full set with book_min less its reduction (lpca): the route to its fourth link.**
- **Levels 0–4:** the comparison, the link, the addition, the row and the multiply (10 x at level 0).
- **Level 5:**
  - LAWNEAR added x² to the near misses (fit 3, against 5–6 for the 12 by fit);
  - the subgoal score singled it out (key (1, 0), mod(g,N)), and it was practised as `T.val.126.tgt.XN`;
  - its rest (`rest.126@XN`) was posed, and not solved: the probe stops short of cost 5, and the deep probe's focus
    held `T.lt.pair.AX` or nothing;
  - the restriction x² < 2N was practised as `T.target.rst5.XN` (x², then the link).
- **Levels 6–8:** values that are x² less multiples of N (`T.target.rst5.XN B[B[T.red.pair.AN]]`):
  - right on all 16 practice probes, so the small target's search had states right on all 16 probes from cost 1;
  - none accepted at width 3;
  - each level's subgoal was one of them (key (1, 58) at level 6: mod(g,N), and by table a function of g alone).
- **The end:** the deadline, 2,400 s at level 8; no round.

**B2lp:** the five pieces in lp's order. The subgoal score singled out values from level 3 and LAWNEAR acted from
level 6. 38 duplicates. The deadline at level 7 (2,632 s). No round.

**B2rd, B2 with REST_DEEP: the round at level 5 (625 s).**
- Levels 0–4 as B2's.
- **Level 5:**
  - x² posed by LAWNEAR and singled out by the subgoal score, as in B2;
  - REST_DEEP probed its rest at 10M states over the composition vocabulary and found `LDA.N P[BRG.X B[T.red.pair.AN]]`
    and `CLA P[BRG.X B[T.red.pair.AN]]` at cost 5, accepted;
  - lpca practised it as `T.target.rest.126.XN`, beside x² (`T.val.126.tgt.XN`), the restriction x² < 2N
    (`T.target.rst5.XN`) and one more value.
- **The small target** came at once, at cost 3: `T.val.126.tgt.XN STA.X T.target.rest.126.XN` and
  `T.target.rst5.XN STA.X T.target.rest.126.XN`. Every listed program is right at full size on the 400 answer rows.
- **The ladders** (`logs/ladder_c2_b2_rd.txt`): both programs 768/768 on all 7 rungs of depth_t and of depth_ood_n_t
  (T = 1 to 64), 28 of 28.

**B2lp_rd, B2lp with REST_DEEP: no round, for a reason of its own.**
- In lp's order the multiply came at level 5 and the restriction x² < 2N at level 6.
- From level 3 the subgoal score singled out a value each level, and REST_DEEP probed that value's rest five times, at
  levels 3–7. It found nothing. The deadline came at level 7 (2,766 s).
- **The reason** (checked offline, `logs/lp_link_check.txt`):
  - lp practised the link in A against N as `T.red.pair.AY B2.4`, with B2.4 = `T.lt.pair.XN SEL.S`. That comparison
    loads X into A, and X is the register the long division shifts;
  - so the 5-token reduction over it gives x mod N on 366 of 4,719 rows (x < 1,000, N < 100);
  - over lpca's link, `STA.S T.lt.pair.AN SEL.S`, on all 4,719.

  Under lp the reduction is not 5 tokens away.

**One flag off** (B2rd's arm, to level 6; `records/c2_b2_rd_no_*`):

| off | the round | how far |
|---|---|---|
| REST_DEEP (= B2) | no | 8 levels, to the 2,400 s deadline |
| SPELL | no | to the cap (962 s): REST_DEEP's probe of the rest from x² at level 5 finds nothing |
| LAWNEAR | no | to the cap (1,052 s): x² is never a near miss; REST_DEEP probed another value's rest at level 6 and found nothing |
| SUBGOAL | no | to the cap (1,230 s): no subgoal, so REST_DEEP has nothing to aim at |
| CLEAN_VALUES | **yes**, level 5 (1,141 s) | `T.val.142.tgt.XN STA.X T.target.rest.142.XN`. With the dirty values back, each level is slower |
| ROWCHECK | **yes**, level 5 (524 s) | B2rd's own programs |
| RESTRICT_ROWS | **yes**, level 5 (604 s) | only the x² spelling (no rst5) |
| TIEBREAK | identical by construction | it never acted in B2rd |

Every one of these rounds is right on every rung of both ladders (`logs/ladder_c2_loo.txt`: 70 of 70).

### 6. Predictions against outcomes

| what | prediction | confidence | outcome |
|---|---|---|---|
| the guess's test | (a) or (b): the reduction's state tested in another spelling | medium | **right**: (b) |
| | (c) or (d) | low | not so |
| the replays | round 1's records, apart from times | high | **right**: all three |
| B1 | x² < N posed from level 0 | high | **right** (rst4, 15% of the rows) |
| | not learnable: it needs the multiply | high | **right** |
| | none of the six pieces; no small target, no round | high | **right** |
| B1lp | the same | high | **right** |
| B2 | the five pieces at levels 0–4 | medium-high | **right** |
| | after the multiply, x² among the near misses | high, given the multiply | **right**: level 5, by LAWNEAR |
| | x² the subgoal, practised | medium-high | **right** |
| | its rest posed | certain, given the line above | **right** |
| | that rest learned with SPELL | medium | **wrong**: the probes never reached it (REST_DEEP followed) |
| | the round | medium-low | **no** |
| B2lp | the same, less likely | low-medium | **no** |
| the tie-break | singles out nothing | medium-high | it never ran (§7) |
| the row check | changes statuses, nothing on the route | medium | **right**: the round comes without it, at the same level |
| SPELL's check (addendum 1) | the reduction accepted at level 5 of the test's setup | certain | **right** |
| B2rd (addendum 2) | levels 0–5 as B2's, to REST_DEEP's probe | high | **right** |
| | REST_DEEP finds a reduction, accepted | medium-high | **right** |
| | it is practised | high | **right** |
| | the round, by level 6 | medium | **right**: level 5 |
| | right on every rung of both ladders | high | **right**: 28 of 28 |
| B2lp_rd | the same route later | low-medium | **no**: lp's link in A against N reads X, so the reduction is not 5 tokens away |
| one flag off | REST_DEEP, LAWNEAR, SUBGOAL or SPELL: no round | high | **right**: all four, no round |
| | CLEAN_VALUES, ROWCHECK, TIEBREAK or RESTRICT_ROWS: the round still comes | medium | **right**: CLEAN_VALUES, ROWCHECK and RESTRICT_ROWS off, the round at level 5; TIEBREAK never acted |

### 7. Dead ends, and how hard each was tried

| what | how hard | where it stopped |
|---|---|---|
| bar 3's full set alone (B1, B1lp) | 3 and 10 levels (72 s, 270 s): every flag on, six restrictions a level, x² < N among them | no piece: nothing poses the comparison, the link, the addition or the row, and the multiply is 9 tokens over the primitives |
| bar 3's full set with book_min less its reduction, without REST_DEEP (B2, B2lp) | 9 and 8 levels (0–8, 0–7), to the 2,400 s deadline | x² practised and its rest posed (B2, level 5), but never solved: the probe stops short of cost 5, and the deep probe's focus missed the link; then values that are x² less multiples of N, right on every practice probe and wrong at width 3 |
| the tie-break (TIEBREAK) | on in every arm | never ran: the practised subgoal's rest is posed again only under a new value's name. A port limit; in B2rd the round came first |
| SUBGOAL's deeper probe at the subgoal itself | on in every arm | never fired: the subgoals were values, solved at once by their own prefix |
| restrictions measured on the target's 8-probe draw (round 1) | fixed by RESTRICT_ROWS | x² < N posed; learnable only with the multiply, which only the book supplies |

### 8. What is left

- **The five pieces below the reduction still need the book:** the comparison, the link, the addition, the row and the
  multiply. No bar 3 source poses a function other than the target's, and alone the multiply's specification (x² < N)
  cannot be learned without the row.
- **Under lp the route needs a clean link.** lp practised the link in A against N as a composite that reads X (§5),
  so no 5-token reduction exists in its library.
- **The tie-break needs a name for the practised subgoal's value that survives the level**, for example matching the
  rest's value rather than its key.
- **SPELL is in this study's copy only.** The committed stage-3 search tests one spelling a goal state. Any search here
  that accepts on more rows than its probes has the same blind spot.
- **REST_DEEP's 10M states and SPELL's 16 spellings were not tuned.** Each was set once, before its arms.
- **Not run:** the generic-instruction tiny core; more seeds; TIEBREAK off (identical by construction, §5).

### 9. Cost (`cost.py`; `logs/cost_c2.txt`, `logs/watch_c2_*.txt`)

| app | what | billed |
|---|---|---|
| ap-DPry7prHfDelGRUXLYVjc1, ap-wnJWCOromJdTZcgvKQqdPw | spell_test, spell_fix | $0.32 |
| ap-sCIzoY3CrMShugxjwu1Jyh | lawnear_test | $0.13 |
| ap-VdukrJkH3FphVS1bSUmMuk | the three replays, B1, B1lp, B2, B2lp | $11.78 |
| ap-g0qTjnbvnmnzRbQq483tNE, ap-NWMwLMOcrtRHgmq6XCUGWS | spell_full, rest_deep_test | $0.62 |
| ap-qZJXhZCdf2hUwpjfbam0zv | B2rd, B2lp_rd | $3.17 so far |
| ap-yflupemrMgMmPD0vbgLDrR, ap-S7H7am4iYYIHLWYv7vYKvB | six flags off | $1.21 so far |
| all | | **$17.23** billed (to 16:00 UTC) |

The hour from 16:00 UTC was not yet billed when this was written. In it ran about 117 B200-minutes: B2lp_rd's last 30,
the four flags-off arms' last 62, and the two later ones' 19, with startup. That is about $12.4 at the first hours'
rate ($6.37 a B200-hour), so **about $29.6 in all**: under the soft $30 and the $40 cap. The watcher's last
estimate (16:41 UTC, as it exited with every app stopped) was $29.37. Every launch passed
`spend.py 4750`. One spend watcher covered every app (stop each at $36, all at $36), and all nine apps are stopped.
Local work ran under an 8 GB memory cap.

### 10. Files and how to rerun

- `PREREG.md`, `PREREG_ADDENDUM.md` (SPELL), `PREREG_ADDENDUM2.md` (REST_DEEP): read-only, hashes in §1.
- `make_copies.py` writes the copies (§2.2): `spcore_c2.py`, `oracle_c2.py`, `core_c2.py`, `climb_c2.py`,
  `clean_c2.py`. `hooks.py` holds the flags' code, `subgoal_c2.py` the score.
- `modal_compose2.py`: the arms (ARMS) and the tests (CHECKS), one B200 each, app and Volume `bar3-compose2`.
- Tests: `spell_test.py` (the guess; with COPY=1 the fix; with FULL=1 the composition vocabulary), `lawnear_test.py`
  (LAWNEAR and SUBGOAL; with REST_DEEP=1, REST_DEEP end to end), `check_subgoal.py`, `restrictions_c2.py`.
- `cost.py` (§9). Round 1's `verify.py` (summary, ladder) and `same.py` serve here too.
- `records/`: every run's log and record, fetched from the Volume. `logs/`: the local tests, the launches, the watcher,
  the ladder check, the copies the main arms ran (`*_main_arms.py.txt`).

```
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/compose2/make_copies.py
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose2.check_subgoal
cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose2/modal_compose2.py --arms spell_test,spell_fix,lawnear_test,spell_full,rest_deep_test
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose2/modal_compose2.py --arms replay_cp_none,replay_cp_book_min,replay_cp_bar3_clean,c2_b1,c2_b1_lp,c2_b2,c2_b2_lp
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose2/modal_compose2.py --arms c2_b2_rd,c2_b2_lp_rd
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose2/modal_compose2.py --arms c2_b2_rd_no_lawnear,c2_b2_rd_no_subgoal,c2_b2_rd_no_spell,c2_b2_rd_no_clean_values,c2_b2_rd_no_rowcheck,c2_b2_rd_no_restrict_rows
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/compose2/modal_compose2.py --mode fetch
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.same bar3/scale/compose/records/cp_book_min.json bar3/scale/compose2/records/replay_cp_book_min.json
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.verify ladder bar3/scale/compose2/records/c2_b2_rd.json "T.val.126.tgt.XN STA.X T.target.rest.126.XN"
```
