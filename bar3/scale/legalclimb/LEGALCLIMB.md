# A legal climb from scratch

*Written 2026-10-26. Thirteen B200 runs on Modal (two tiny tests, eleven arms), about $20 (§7). Code in this folder;
records in `records/`, logs in `logs/`. Nothing was committed.*

<!-- PLAIN -->
## In plain words

**The question.** Test 2 (bar3/scale/legal/LEGAL_TEST.md) showed that the composition route's last two steps work
legally at full size, given a library that a run with Python's answers had built. Test 3 runs the whole climb legally,
from nothing:
- the target is H1's own T = 1 rows, and bar 3's target-derived sources are posed on them at full size;
- every other practice answer comes from the data, an identity or the learner's own routines;
- the routine kept among accepted spellings is the climb's own choice, fewest steps executed, and test 2's identity
  check is a second guard behind its own flag.

Does the round come, and how far down the law book can the legal sources go?

**What came: with the book's five pieces and bar 3's sources, the legal climb gets H1's round every time.** Seven
arms did, all at level 5 and all right on both ladders. The ones without a round lacked a piece or the sources.

| arm | the book | bar 3's sources at full size | the round |
|---|---|---|---|
| `c1` | book_min | none | level 5, 459 s |
| `g0` | the floor: book_min, its reduction cut to one step | none | none in 10 levels |
| `g1` | the floor | values and rests on the answer rows | level 5, 794 s |
| `g1_id` | the floor | the same, the identity check on | level 5, 796 s |
| `g1_b2` | the floor | the same, the vocabulary's bodies up to 2 | level 5, 414 s |
| `g1_s1`, `g1_s2`, `g1_s3` | the floor | the same, seeds 1, 2 and 3 | level 5; 919, 1,161, 1,198 s |
| `g2` | book_min less its reduction | the same, the law view's floor at 4 rows | none in 10 levels |
| `g2_m3` | book_min less its reduction | the same, the floor at 3 rows | level 5, 664 s |
| `g3` | the floor less the row | the same | none in 10 levels: no multiply |

Every round is right on both of H1's ladders: 768 of 768 on all 14 rungs, T = 1 to 64, seen and unseen moduli.

**How the round comes** (the same in the floor's six arms; g2_m3's differences are below):
1. **Levels 0–4, the book's laws:** the comparison, the link, the addition, then the multiply-add row and the chain
   (red on x + d·N, Horner's step) together, then the multiply. Each is answered by a fact, an identity or the
   learner's own routines.
2. **Level 5, bar 3's sources, legally:**
   - the look runs the round's own search on the 780 answer rows to cost 3, and tests 31,000–34,000 distinct values
     with the learner's own chain and comparison;
   - only x² passes, on its 16 exact rows;
   - x² and the rest of the way from it are posed on the answer rows at full size;
   - the rest's reduction, `LDA.N P[BRG.X T.red.x+dn.AN]`, is found at cost 4 by the regular probe.
3. **The round** is 3 tokens over those two routines, `T.val.0.tgt.XN STA.X T.target.rest.0.XN`, found by the
   round's own search.

**What the guards did.**
- Without the identity check (g1), the full-size probe accepted test 2's two look-alike spellings too. The climb's own
  choice still kept the right one (445 steps against 448).
- With the check (g1_id), the look-alikes were rejected before any choice.
- With bodies of at most 2 items (g1_b2), the look-alikes cannot be spelled at all, and the climb is twice as fast:
  414 s instead of about 800.

**How far down the book: book_min less its whole reduction.**
- **The book keeps five laws:** the comparison, the link, the addition, the row and the multiply. None has a legal
  source but the book: the target's rows never show them alone, and nothing holds x² before a multiply exists.
  Without the row (g3) the multiply never came, in 10 levels.
- **The reduction can go entirely, loop and step** (g2_m3: book_min less its reduction). The rest of the way from x²,
  posed on the data's rows, gave a 5-token reduction over the link, `LDA.N P[BRG.X P[T.red.pair.AN]]`, found by
  REST_DEEP's 10M probe in 166 s. The round came at level 5.
- **What that took: the law view's floor at 3 rows.** With the link as its only reduction, the law view sees x² exact
  on 3 of the 780 rows. At the pre-registered floor of 4 (g2), nothing passed in eight looks and no round came. At 3,
  x² was the only one of 21,288 values to pass.
- **With the chain in the book** (red on x + d·N, the reduction's step: g1 and its seeds), x² is exact on 16 rows and
  the regular probe finds a 4-token reduction.
- **Without bar 3's sources the same books get no round** (g0, 10 levels): the round's search has no loops, so the
  reduction's loop over the dividend's digits must already be a routine.

**What it does not show.**
- **The tiny core, not a training run.** The climb runs offline on one B200, as every composition round did. Moving it
  into an sp2-style file is the next step, and test 2's question 3 (keeping rows) comes with it.
- **The judge's question 5** (LEGAL_SOURCES.md §6). The law view here makes its decisions with the learner's routines,
  so it does not need numpy's arithmetic. The choosing rule, lpca, and the search's sizes are still designed.
- **One formula.** The route is x²-shaped (COMPOSE3.md): x³ and 10·x² + 1 need more than this source poses.
- **A thin floor.** The book without its reduction (g2_m3) rests on 3 exact rows of 780, in one run. With the chain
  it rests on 16, in six.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

Each file was frozen (read-only) before the runs it names, its sha256 and the code's hashes in `logs/`:

| file | frozen (PDT) | sha256 | first run it covers |
|---|---|---|---|
| PREREG.md | 20:23:27 | `5114ff12…156a` | the tiny tests, 20:23:32 |
| ADDENDUM1.md | 20:41:08 | `b0fc018f…acb9` | g1_b2 and g1_s1, 20:41:12 |
| ADDENDUM2.md | 21:09:59 | `f781cac6…c2e6` | g2_m3, g1_s2, g1_s3, 21:10:04 |

Each header's time was written a minute or two ahead and is not the freeze time, which the table gives:
PREREG.md's says 20:25, ADDENDUM1.md's "about 20:42", ADDENDUM2.md's 21:10. Two of ADDENDUM1.md's table lines run 123
columns. The frozen files were left as they are. Their texts, unchanged:

<!-- PREREG -->
# A legal climb from scratch: pre-registration

Written 2026-10-26 20:25 PDT, before any run on a GPU, the tiny tests included. LEGALCLIMB.md §1 repeats this text
unchanged and records the time and hash it was frozen with.

## The question

Test 2 (bar3/scale/legal/LEGAL_TEST.md) showed that the composition route's last two steps work legally at full size,
given s4_x2's library: the law view by the learner's own routines picks x², and the rest of the way posed on H1's
answer rows finds the reduction. But that library came from a run with Python's answers. Test 3 runs the whole climb
with legal practice answers only:
- the target is H1's T = 1 rows, and bar 3's target-derived sources are posed on them at full size;
- every other practice answer comes from the data, an identity or the learner's own routines;
- the routine kept among accepted spellings is the climb's own choice (fewest steps executed), and the identity check
  at one more digit is a second guard, behind its own flag.

Does the round come? How far down the book can the legal sources go, and which pieces cannot be posed legally?

## Theory, in brief: what can be posed legally

- **The book's laws are legal** (spcore's law_target): facts of the data's numbers, identities, the learner's routines.
- **Bar 3's target-derived sources, legally, are the values and their rests on the answer rows at full size**
  (bar3/LEGAL_SOURCES.md §3). Restrictions pose nothing on H1, and projections were never learnable: both are left out.
- **The round's pieces, one by one:**
  - the comparison, the link and the addition: only the book poses them. The target's rows never show them alone;
  - the multiply-add row and the multiply: answered legally, round 4's cut of the multiply is the book's row law
    itself, and choosing it from the target's law needs numpy (LEGAL_SOURCES.md §3.13 and §6). No value holds x²
    before a multiply exists, so nothing target-derived poses either. They stay in the book;
  - the reduction: its loop over the dividend's digits is the rest of the way from x², target-derived and legal at
    full size. Its step, Horner's (red on x + d N, the chain), is the book's identity. The legal law test needs the
    chain: with the link alone, x² is exact on 3 of the 780 answer rows, under the floor of 4.
- **So the predicted floor is book_min with its reduction (red on x + o N) swapped for the chain (red on x + d N)**:
  lt, red on the pair, add, the row a + b·c, the multiply, the chain. Bar 3's sources supply the reduction's loop.
- **Without the target-derived sources the floor's book gets no round.** The round's search is straight-line over
  routines, with no loops, so the reduction's loop must already be a routine.

## Design (`make_copies5.py`, `rowclimb.py`, `lclimb.py`)

- **The copies:** round 4's committed copies (sha256 asserted), module paths moved here, 1 + 4 asserted replacements:
  SUBGOAL's score for a full-size target is its own law view; TARGET=rows builds the full-size target; the climb is
  LegalClimb; LOOK_COST, IDCHECK and FULL_BODY are recorded. With no full-size root, LegalClimb is ComposeClimb.
- **The full-size mode in the level loop** (`rowclimb.RowClimbTarget`), each level:
  - the look, at the pose phase: the round search's vocabulary on all 780 answer rows, every state of cost ≤ 3 kept;
    each distinct value of A, X, Y and S tested by the law view by routines (the learner's chain on x + d N, else
    on the pair, and its comparison; digit equality; at least 4 rows where the chain is exact); up to 4 that pass;
  - each one's value, if clean and neither an input nor constant, posed on the target's root (X = x, N = N), and its
    rest of the way on a root of its own (X = the value, N = N, the row's answer); both at full size, 8 probe rows;
  - full-size roots probed apart (`lclimb.LegalClimb`): their own setting and the cached full-size vocabulary
    (bodies up to 3, signatures kept across levels); no primitives-alone probe; no regular deep probe; REST_DEEP's
    10M probe for the subgoal's rest when the regular probe misses it;
  - acceptance on all 780 answer rows. With IDCHECK, a rest must also pass the identity check: red(r + o' N') = r,
    with N' = 10 N + d, o' = 10 o + e (digit moves of the rows' own numbers), r = x + k N (k additions of N), o' N'
    by the learner's multiply and the sum by its addition;
  - SUBGOAL: the one value that passed, if exactly one did and it was posed this level;
  - the routine kept: the climb's own choice, on the root's own setting;
  - the round: the round's own search (search_round, accepted on the 780 rows).
- **Settings, every arm:** lpca, K 4, NP 8, deep probes over whole concepts (DEEP_BY bindings, DEEP_FOCUS closure),
  probe 1M, deep 5.8M (8 roots), SUBGOAL's and REST_DEEP's 10M, a search's limit 900 s, LEVELS 10, SECONDS 2,700,
  SPELL 16, ANSWERS 780, ROWS 4,096, ROUND 4M; LOOK_COST 3, LAW_NEAR 4, MIN_ROWS 4.
- **The round's check:** both ladders, T = 1 to 64, seen and unseen moduli (compose3/verify3.py, locally).

## Runs

**Tiny tests first**, one B200 each, in parallel (`resume.py`, MODE=level): s4_x2's climb resumed at the start of
its level 5 (library, routines, practised problems as recorded), with TARGET=rows and IDCHECK, one level:
- `rs_b3`: the full-size vocabulary's bodies up to 3, as the arms;
- `rs_b2`: bodies up to 2.

**Then the arms**, at most 5 B200s at once, spawned in this order (g2 queues):

| arm | book | sources |
|---|---|---|
| `g1` | the floor: LAWS lt;red;add;mul;acc.mul, PATTERNS pair;x+dn;x,o;o,x,d | law, values and rests at full size |
| `g1_id` | the floor | as g1, IDCHECK on |
| `g0` | the floor | the law book alone (TARGET round) |
| `c1` | book_min: PATTERNS pair;x+on;x,o;o,x,d | the law book alone (TARGET round) |
| `g3` | the floor less the row: LAWS lt;red;add;mul, PATTERNS pair;x+dn;x,o | as g1 |
| `g2` | book_min less its reduction, no chain: PATTERNS pair;x,o;o,x,d | as g1 |

A round anywhere is checked on both ladders.

## Seen before this was written (local, CPU)

- Flags off: two-level tiny runs with TARGET round and TARGET small give round 4's records exactly, apart from times.
- TARGET=rows at tiny budgets: three levels without error. The look waits for a chain.
- The pieces on s4_x2's level-5 state (`resume.py`, MODE=pieces):
  - the chain found (`T.red.Bx+d.AN`); the pose (val.0 on tgt@XN, its rest on rest.0@XN);
  - the value's acceptor (x² yes, x no);
  - SUBGOAL val.0; the climb's routine choice among test 2's four spellings, `LDA.N P[BRG.X T.red.Bx+d.AN]`;
  - the vocabulary's cache.
- The identity check changed twice before this was frozen. With r = x it passed `P[INC BRG.X T.red.Bx+d.AN]`. With
  r = x + k N it still passed it. With o moved up one digit too (dividends of 15–16 digits, as x² for 24-bit N) it
  rejects both look-alikes and passes the two right spellings.
- The counts: on the 780 answer rows x² < 2 N on 3 rows, x² < 11 N on 16.

## Predictions

| what | prediction | confidence |
|---|---|---|
| rs_b3, rs_b2 | x² alone passes; val.0 at cost 3; its rest at cost 4 at 1M; the round at level 5 | medium-high |
| rs_b3, rs_b2: the round's ladders | 768/768 on all 14 rungs | medium-high |
| rs_b3: the full-size vocabulary | 10–15 minutes to build; rs_b2 under 2 minutes | medium |
| c1 | the round at level 5, by the round's own search | high |
| g0 | no round: the six pieces practised by level 5, then nothing reduces | medium-high |
| g1 | the six pieces at levels 0–4; x² posed at level 5; its rest found; the round at level 5 or 6 | medium |
| g1: the round's ladders | 768/768 on all 14 rungs | medium-high, given a round |
| g1_id | as g1; the identity check rejects any look-alike spelling it meets | medium |
| g2 | the look never passes x² (3 exact rows); no round | high |
| g3 | no multiply, so no x² and no round | high |
| the floor | book_min with its reduction swapped for the chain (below) | medium |
| money | about $20–28 in all | medium |

The tiny tests' round is `T.val.0.tgt.XN STA.X T.target.rest.0.XN`. The floor, in full: book_min with its reduction
(red on x + o N) swapped for the chain (red on x + d N); the reduction's loop comes from bar 3's sources, legally.

## Money

Soft $40, cap $60. `integration/ablation_s3/spend.py 4750` runs before each launch. One spend watcher
(bar3/scale/digits/watch_spend.py) for this study's apps, its total the cap less one hour of the containers' cost: 5
B200s at $6.37 an hour, so it stops everything at $28. Every app is stopped when its runs end.
<!-- /PREREG -->

<!-- ADDENDUM1 -->
# A legal climb from scratch: addendum 1

Written 2026-10-26 at about 20:42 PDT, before the two arms it adds were launched. LEGALCLIMB.md §1 repeats it unchanged
and records the time and hash it was frozen with.

## What changed

- **The budget** (the coordinator, relaying the user, at about 20:38 PDT): soft $100, cap $150, from $40 and $60. The
  spend watcher's stop-all moved to $110 (the cap less one hour of six B200s), without stopping the running arms.
- **Two arms added**, in an app of their own, one B200 at a time (six at most, with the first app's five):
  - `g1_b2`: g1 with the full-size vocabulary's bodies up to 2 (FULL_BODY=2). The tiny test rs_b2 built 6,386 item
    classes in 46 s, against rs_b3's 285,819 in 519 s, and took the same route. If the climb gets its round this way,
    this is the configuration a training file can afford.
  - `g1_s1`: g1 with the climb's seed 1 (SEED=1: the practice roots' draws), to see whether the route depends on one
    draw.
- **The code they need** (hashes in `logs/addendum1_sha256.txt`):
  - `core_l.py`: one replacement more (make_copies5.py), SEED as the climb's seed. It is inert unless SEED is set.
  - `modal_lclimb.py`: the two arms; MAX read from LEGALCLIMB_MAX (1 for the added arms' app).

  The running arms run the code frozen with PREREG.md: their image was built when they were launched.

## Seen before this was written

- The tiny tests: rs_b2 and rs_b3 got the round at level 5, right on both ladders.
- The first arms' early levels: five running, two of them posing two candidates at level 5 (the look's value and its
  rest), one at level 6 posing 69. Nothing else.

## Predictions

| arm | prediction | confidence |
|---|---|---|
| `g1_b2` | as g1: the round at level 5 or 6; the full-size vocabulary under 2 minutes; the run under 20 minutes | medium |
| `g1_s1` | as g1: the round at level 5 or 6 (the draws change which bindings are practised, not which concepts) | medium |
| both, given a round | right on both ladders, 768/768 on all 14 rungs | medium-high |
<!-- /ADDENDUM1 -->

<!-- ADDENDUM2 -->
# A legal climb from scratch: addendum 2

Written 2026-10-26 21:10 PDT, before the three arms it adds were launched. LEGALCLIMB.md §1 repeats it unchanged and
records the time and hash it was frozen with.

## Why

The first eight arms ended as pre-registered, but for g3, still running (no multiply by level 7):
- c1 got its round at level 5;
- g1, g1_id, g1_b2 and g1_s1 got theirs at level 5, `T.val.0.tgt.XN STA.X T.target.rest.0.XN`, right on both ladders;
- g0 and g2 got none, by the level cap.

g2 failed where predicted: with the link as its only reduction, the law view saw x² exact on 3 of the 780 rows, under
its floor of 4, and passed nothing in eight looks. Two questions the budget now allows:
- **How far down the book?** Is the chain needed only by the floor? If the law view accepts 3 rows, the reduction's
  rest over the link costs 5 tokens (`LDA.N P[BRG.X B[T.red.pair.AN]]`, round 2's), beyond the regular probe at full
  size and inside REST_DEEP's 10M only if that probe has time to stream its last level.
- **Is the route robust?** Two seeds so far, s0 and s1; two more make four.

## The arms (an app of their own, three B200s at once)

| arm | what |
|---|---|
| `g2_m3` | g2 (no chain) with the law view's floor at 3 (MIN_ROWS=3); SECONDS 3,600, SEARCH_S 2,400 (below) |
| `g1_s2`, `g1_s3` | g1 with the climb's seeds 2 and 3 |

g2_m3's longer limits (a search's 2,400 s, the run's 3,600 s) give REST_DEEP's 10M probe the time to stream its last
level at full size.

Code (hashes in `logs/addendum2_sha256.txt`): `rowclimb.py` takes `min_rows` (rowtarget's MIN_ROWS, read at each law
test); `core_l.py` passes MIN_ROWS (one replacement changed in make_copies5.py; inert unless set); `modal_lclimb.py`
has the three arms. The running arm, g3, runs the code it was launched with.

## Seen before this was written

- The eight arms' results above, and g3 to level 7.
- Locally on s4_x2's level-5 state: with the link, x² has 3 exact rows and passes at floor 3, not at 4; x² with its
  lowest digit changed fails both. With the chain: 16 rows, passes.

## Predictions

| arm | prediction | confidence |
|---|---|---|
| `g2_m3` | x² passes at level 5, alone, on 3 rows; the 1M probe misses its rest; REST_DEEP finds it | low-medium |
| `g2_m3`, if REST_DEEP finds nothing | no round (below) | medium, given the miss |
| `g1_s2`, `g1_s3` | as g1: the round at level 5, right on both ladders | medium-high |

g2_m3 in full: REST_DEEP's 10M probe finds a 5-token reduction over the link and the round comes at level 5. If it
finds nothing, no round: later looks pass x² again, but a practised value is no longer a subgoal, so it gets no
deeper probe.
<!-- /ADDENDUM2 -->

### 2. The full-size mode in the climb

**The copies** (`make_copies5.py`). Round 4's committed copies (`bar3/scale/slice/`, sha256 asserted), module paths
moved here, then this study's replacements, each asserted to match once:

| file | replacements |
|---|---|
| `hooks.py` | 1: SUBGOAL's score, for a full-size target, is the target's own (`RowClimbTarget.subgoal_scores`) |
| `core_l.py` | 5: TARGET=rows; the climb is `LegalClimb`; LOOK_COST, IDCHECK, FULL_BODY, MIN_ROWS; SEED (addenda) |
| `spcore_l.py`, `climb_l.py`, `oracle_l.py`, `clean_l.py` | none: paths moved |

**The target** (`rowclimb.RowClimbTarget`): test 2's `rowtarget.RowTarget` with the hooks SelfClimb calls on its
target, in place of the small target's:

| SelfClimb calls | the small target (round 4) | the full-size target (here) |
|---|---|---|
| `search`, pose phase | its own search, practice widths; LAWNEAR by numpy | the round's vocabulary, 780 rows, cost 3 |
| the law view | numpy or torch on g, X and N | the learner's chain and comparison, digit equality |
| `pose` | values, rests, restrictions, projections | the passing values and their rests, full size |
| `acceptor` | 60 rows a verify width, by Python | the 780 answer rows; IDCHECK's identity instances |
| `subgoal_scores` | numpy's law and table views | the one value that passed, posed this level |
| `search` at the target phase | the small target itself | nothing: the round's own search follows |

**The climb** (`lclimb.LegalClimb`). SelfClimb groups roots by their probes' shapes, so full-size roots already form
groups of their own. A full-size group is probed on its own setting over the target's cached full-size vocabulary,
with no primitives-alone probe. The regular deep phase leaves it out; its deep probe is REST_DEEP's. Its routine is
the climb's own choice (inputs kept, registers touched, size, steps executed), on its own setting. With no full-size
root every method is ComposeClimb's.

**The vocabulary's cache.** enumerate_items keeps every item's signature on fixed test machines. A new library runs
only its new items, and the classes are rebuilt only when the routines change.

**The identity check** (IDCHECK), built from the learner's routines and the rows' own numbers only:
- N' = 10·N + d and o' = 10·o + e: digit moves (o is another row's x);
- r = x + k·N: k additions of N, by the learner's addition;
- r + o'·N': the learner's multiply, then its addition.

A reduction found from the data must also turn r + o'·N' into r on all 780 instances.

### 3. Checks before the runs (local, CPU)

- **Flags off.** Two-level tiny runs of round 4's core and this one, with TARGET round and with TARGET small (bar 3's
  sources at the practice widths, every round-2 flag on), give the same records apart from times (`same.py`).
- **TARGET=rows, tiny.** Three levels without error, the look waiting for a chain (peak 2.7 GB, under a 3 GB cap).
- **The pieces on s4_x2's level-5 state** (`resume.py`, MODE=pieces): the chain found; the pose; the value's acceptor;
  SUBGOAL; the climb's choice among test 2's four spellings (`LDA.N P[BRG.X T.red.Bx+d.AN]`); the cache.
- **The identity check needed two fixes** before it told test 2's spellings apart (all four tested on 780 instances):

| instances | `P[INC …]` | `P[… INC]` | `LDA.N P[…]` | `CLA P[…]` |
|---|---|---|---|---|
| r = x, o a row's x | passes | fails | passes | passes |
| r = x + k N | passes | fails | passes | passes |
| r = x + k N, o moved up one digit (frozen) | fails | fails | passes | passes |

  The first look-alike fails only when the dividend fills 15–16 digits over an 8-digit modulus, as x² does for 24-bit
  N. With o from the data's 7-digit x the instances stop at 13–14 digits.

### 4. The tiny tests (one B200 each)

s4_x2's climb resumed at the start of its level 5, one level, TARGET=rows, IDCHECK on (`resume.py`, MODE=level):

| | rs_b2 (bodies up to 2) | rs_b3 (bodies up to 3, as the arms) |
|---|---|---|
| the look | 41,014 values, x² alone passes (16 exact rows), 7 s | the same |
| the full-size vocabulary | 6,386 classes, 46 s | 285,819 classes, 519 s |
| the probe (val.0 cost 3, rest.0 cost 4) | 48 s | 73 s |
| the rest's accepted spellings | `LDA.N P[…]`, `CLA P[…]` | the same; IDCHECK rejected both INC spellings |
| the routine kept | `LDA.N P[BRG.X T.red.Bx+d.AN]` | the same |
| the round, level 5 | `T.val.0.tgt.XN STA.X T.target.rest.0.XN`, 106 s | the same, 601 s |
| both ladders, T = 1 to 64 | 768/768 on all 14 rungs | 768/768 on all 14 rungs |

### 5. The arms

**The levels** (the floor's book; c1 the same until level 5, where it practises its reduction instead):

| level | practised | the look |
|---|---|---|
| 0 | lt on the pair (the comparison) | no chain or comparison yet |
| 1 | red on the pair (the link) | none yet |
| 2 | add on the pair | the link as its chain: 9,154 values, none passes |
| 3 | the row a + b·c; red on x + d·N (the chain); lt on x + d·N | the link: 12,855 values, none |
| 4 | the multiply on (x, o) | the chain: 29,757–31,450 values, none (no multiply yet) |
| 5 | x² (`LDA.X STA.Y T.mul.x,o.XY`) and its rest | the chain: 30,932–33,749 values, x² alone (16 rows) |

The look's time: under 0.5 s a level until the chain exists, then about 4 s.

**Each arm's level 5** (the floor and bar 3's sources):

| | g1 | g1_id | g1_b2 | g1_s1 |
|---|---|---|---|---|
| level 5 starts | 286 s | 284 s | 332 s | 302 s |
| the full-size vocabulary | 240,997 classes, 438 s | 240,997, 443 s | 5,575, 45 s | 229,080, 544 s |
| the full-size probe's search | 66 s | 68 s | 35 s | 66 s |
| the rest's spellings accepted | test 2's four | `LDA.N P[…]`, `CLA P[…]` | `LDA.N P[…]`, `CLA P[…]` | test 2's four |
| the routine kept | `LDA.N P[BRG.X T.red.x+dn.AN]` | the same | the same | the same |
| the round (cost 3, by its own search) | 794 s | 796 s | 414 s | 919 s |
| both ladders | 768/768 × 14 | 768/768 × 14 | 768/768 × 14 | 768/768 × 14 |

**The added arms' level 5** (addendum 2; g1_s2 and g1_s3 the floor, g2_m3 book_min less its reduction):

| | g1_s2 | g1_s3 | g2_m3 |
|---|---|---|---|
| level 5 starts | 371 s | 373 s | 319 s |
| the look | 32,579 values, x² alone (16 rows) | 32,664, x² alone (16) | the link: 21,288, x² alone (3) |
| the full-size vocabulary | 238,105 classes, 704 s | 238,335, 735 s | 174,373, 122 s |
| the rest, regular 1M probe | cost 4, the four of test 2 | the same | missed |
| REST_DEEP's 10M probe | not needed | not needed | cost 5 in 166 s: `P[link]` and `B[link]` in the loop |
| the routine kept | `LDA.N P[BRG.X T.red.x+dn.AN]` | the same | `LDA.N P[BRG.X P[T.red.pair.AN]]` |
| the round | 1,161 s | 1,198 s | 664 s |
| both ladders | 768/768 × 14 | 768/768 × 14 | 768/768 × 14 |

g2_m3's kept routine runs sixteen conditional subtractions a digit where ten suffice, and is still exact. The climb's
choice counts steps at the practice width, where a P-loop is the shorter.

**The controls.**
- **c1, book_min alone** (TARGET round): its reduction (red on x + o·N) at level 5, and the round at level 5 (459 s):
  `LDA.X STA.Y T.mul.x,o.YX STA.X T.red.x+on.XN`, 768/768 on all 14 rungs.
- **g0, the floor alone**: the floor's concepts all practised by level 4. From level 5 on, it practised only other
  bindings of the same laws, and no round came by the level cap (10 levels, 581 s). Its round search, straight-line
  over routines and then by calls, cannot spell a loop.
- **g2, without the chain**: the same climb as c1 to level 4. Its look used the link: 9,154 to 44,724 values in eight
  looks (levels 2–9), none passed, because x² is exact on 3 of the 780 rows and the floor is 4. No round by the level
  cap (549 s).
- **g3, without the row**: the multiply was unsolved at every level (12 unsolved, its bindings, each level). Its look
  passed nothing at levels 2–9, and no round came by the level cap (2,492 s).

### 6. Predictions against outcomes

| prediction | outcome | |
|---|---|---|
| rs_b3, rs_b2: x² alone passes; val.0 cost 3; its rest cost 4 at 1M; the round at level 5 | so | held |
| rs_b3, rs_b2: both ladders 768/768 on all 14 rungs | so | held |
| rs_b3: the vocabulary in 10–15 minutes; rs_b2 under 2 | 8.7 minutes; 46 s | held, rs_b3 faster |
| c1: the round at level 5, by its own search | so | held |
| g0: no round; the six pieces by level 5, then nothing reduces | so (the pieces by level 4) | held |
| g1: the pieces at levels 0–4; x² at level 5; its rest; the round at level 5 or 6 | level 5 | held |
| g1: both ladders 768/768 | so | held |
| g1_id: as g1; the identity check rejects any look-alike it meets | so: both INC spellings rejected | held |
| g2: the look never passes x² (3 exact rows); no round | so: 8 looks, nothing passed | held |
| g3: no multiply, so no x² and no round | so | held |
| the floor: book_min with its reduction swapped for the chain | so at the floor of 4; at 3 the chain goes too | held |
| money: about $20–28 in all | about $20 | held |
| addendum 1, g1_b2: as g1; the vocabulary under 2 minutes; the run under 20 | level 5; 45 s; 414 s | held |
| addendum 1, g1_s1: as g1 | level 5 | held |
| addendum 1, both: both ladders | so | held |
| addendum 2, g2_m3: x² alone on 3 rows; the 1M probe misses; REST_DEEP finds it | so: cost 5, 166 s | held |
| addendum 2, g1_s2 and g1_s3: as g1, both ladders | so | held |

Every pre-registered outcome held. One time estimate fell below its range: rs_b3's vocabulary took 8.7 minutes against
the predicted 10–15.

### 7. Cost

- **Billed** (`cost.py`, at 04:32 UTC): $12.47 through the 03:00 UTC hour. That is the tiny tests $1.45, the six
  arms $9.00 and addendum 1's arms $2.01. The 04:00 UTC hour (g3's last 16 minutes and addendum 2's three arms) was
  not in the report yet.
- **By the runs' own times:** 10,733 s of module time on B200s (179 minutes) at $6.37 an hour is $19.0, plus the
  containers' start-up. So about $20 in all.
- **The budget:** soft $40 and cap $60, raised during the runs to soft $100 and cap $150. The spend watcher's
  stop-all was $28, then $110; it never acted. The month's bill was $3,812.93 at 04:32 UTC, under the stop of $4,750.
- **Apps:** all four stopped (the tiny tests', the arms', and each addendum's), and the fetches' too.

### 8. What it shows, and what it does not

- **The legal climb gets the round from nothing.** Every practice answer in g1's climb is a fact, an identity, a data
  row or a routine's output, and every decision on a value is made by the learner's own routines. The library the
  round uses is the climb's own.
- **The floor, legally:** book_min less its reduction (g2_m3, the law view's floor at 3 rows).
  - The comparison, the link, the addition, the row and the multiply have no legal source but the book.
  - Bar 3's rest of the way, posed on the data's rows, supplies the whole reduction, loop and step, over the link.
  - At the pre-registered floor of 4 rows, the reduction's step stays in the book: the chain, red on x + d·N. With it,
    six runs of six got the round.
- **The spelling guards both work.** The climb's own choice kept the right spelling even when the look-alikes were
  accepted. The identity check removed them before any choice. Bodies of at most 2 items never spell them, and halve
  the climb's time.
- **Not shown:**
  - the climb inside a training run (an sp2-style file: rows kept in buffers, judge question 3);
  - other formulas (the route is x²-shaped);
  - the choosing rule (lpca) and the search's sizes, which stay designed.
- **Cost of full size.** With bodies of at most 3 items, building the full-size vocabulary is most of the climb's
  time: 438–735 s of 794–1,198 s in the floor's arms. With at most 2 it is 45 s. A training file would want the
  latter.

### 9. Files and how to rerun

- `make_copies5.py`: the copies (`spcore_l.py`, `climb_l.py`, `core_l.py`, `hooks.py`, `oracle_l.py`, `clean_l.py`).
- `rowclimb.py`: the full-size target. `lclimb.py`: the climb.
- `resume.py`: the tiny tests (MODE=pieces local, MODE=level on a GPU). `same.py`: the flags-off check.
- `modal_lclimb.py`: the launcher (Volume `bar3-legalclimb`). `cost.py`: §7.
- `PREREG.md`, `ADDENDUM1.md`, `ADDENDUM2.md`: frozen. `logs/*_sha256.txt`: their hashes.
- `records/<arm>.json`, `records/<arm>.log`: every run. `logs/ladder_<arm>.txt`: the rounds on both ladders.
- `logs/*_app.log`: the apps' logs as captured; `logs/watch_*.txt`: the spend watchers.

From `cpu_first`, with `PY=~/github/sotaku/venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1` and `H1_DATA` exported:

```bash
$PY -m bar3.scale.legalclimb.make_copies5
MODE=pieces $PY -m bar3.scale.legalclimb.resume
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach bar3/scale/legalclimb/modal_lclimb.py --arms g1,g1_id,g0,c1,g3,g2
$PY -m modal run bar3/scale/legalclimb/modal_lclimb.py --mode fetch
$PY -m bar3.scale.compose3.verify3 bar3/scale/legalclimb/records/g1.json "T.val.0.tgt.XN STA.X T.target.rest.0.XN"
$PY bar3/scale/legalclimb/cost.py
```
