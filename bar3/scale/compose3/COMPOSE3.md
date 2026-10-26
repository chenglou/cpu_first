# Composition round 3: the row, and how general round 2's route is

<!-- PLAIN -->
## In plain words

**The questions.** Round 2 (bar3/scale/compose2/COMPOSE2.md) got H1's round with book_min less its reduction: bar 3's
own sources supplied the last law. Round 3 asks two things:
1. With the book less its reduction and less its multiply-add row, can bar 3's pieces pose the row, or anything that
   gets the multiply?
2. Does round 2's winning setup, unchanged, get the round for targets no rule was written for: x³ mod N and
   10·x² + 1 mod N (SP_BREADTH.md's cube and tenrho), at the tiny core's widths?

**What came: no round in either part, as pre-registered.**

| arm | what | the round |
|---|---|---|
| R1 | book_min less its reduction and its row, with round 2's full set (REST_DEEP and the new TIEFOLLOW too) | no: the comparison, the link and the addition, then never the multiply (10 levels) |
| G3 | round 2's B2rd unchanged, on x³ mod N | no: the five pieces at levels 0–4, then nothing the route can use (7 levels) |
| G10 | round 2's B2rd unchanged, on 10·x² + 1 mod N | no: the same, plus two subgoals by table alone whose rests REST_DEEP could not solve |
| the replays | round 1's book_min and no practice; round 2's B2rd | their records again, apart from times, B2rd's round included |

**Why the row cannot be replaced here.**
- Without the row, the multiply's program is `CLA W[LDH.Y CLK P[MUL.X INC] INC SHL.X]`: 9 tokens over the primitives.
  R1's deep probes on it held up to cost 4 and streamed cost 5 (about 17.5M states each), and found nothing.
- **Bar 3's sources pose only from the target's rows.**
  - The rows where x² < N do pose the square (round 2's RESTRICT_ROWS). But at the acceptance width x' reaches 31,
    so only the two-digit multiply passes.
  - The cheapest program on its one-digit practice probes is a 6-token row with its multiplier digit taken from X,
    which does not compose into the multiply.
  - No value holds x² before a multiply exists, so the law view finds nothing (no LAWNEAR state in all 10 levels).
- **Bar 3's invention of multiply-accumulate (CARRY.md §12) does not carry over to this machine.**
  - It widened registers its problems pin, and here no register is pinned.
  - Its loop-view score loops a step a register's number of times, and here loops run a fixed number of times.
  - Its conjunctions isolated a digit, and here no restriction term can.
  - What does carry over, the deeper probe, was already on.
  - And on this machine the one-place multiply-accumulate is a primitive (`MUL`): what the book supplies is the loop
    over the places, which needs the operands' layout no target row shows.

**Why round 2's route is x²-shaped.** The route needs a value that is one law of depth 1 from the target (over g, X
and N), cheap enough for the look's held levels (cost ≤ 3), and inside the practice registers:
- **x² mod N:** x² is all three (`LDA.X STA.Y T.mul.x,o.XY`, cost 3).
- **x³ mod N:** has no such value:
  - x³ costs about 8 and overflows the registers;
  - x² is two laws away, and the rest from x² would need x, which a rest's inputs do not hold.

  G3: no LAWNEAR state, no subgoal.
- **10·x² + 1 mod N:** its values cost 5 or more, and overflow at width 2. G10: no LAWNEAR state; the subgoal score
  singled out two values by table alone, two rows saved each, not pieces.

So LAWNEAR, as built (depth 1, no constants, the look's reach), is shaped to x². SPELL is a property of the search and
general, and so is REST_DEEP's mechanism. But both act only after a value one law from the target is posed.

**The tie-break now follows the subgoal's value** (TIEFOLLOW, its own flag). On round 2's library, with x² practised
and posed again under a new name, it finds the rest of the way from it. The tie-break by name finds nothing. It never
acted in an arm: R1's only subgoal came at its last level, and the other arms ran without the flag.

**What it means.**
- **The book's last law is replaceable by bar 3's sources, and its row is not.** The difference is what the target's
  rows show:
  - the reduction is the last step of the target, so a value one law from the target exposes it;
  - the row is buried two loops deep inside a piece, the multiply, that the target's rows never show alone at the
    acceptance width.
- **The route found in round 2 works where the target is one cheap law from a value.** On the family's other targets
  it needs what bar 3's score already has and the port left out:
  - deeper laws: the cube is mod(mul(g, X), N) from g = x²;
  - constants, which bar 3 takes from the data;
  - a reach past cost 3;
  - and for the cube, x kept beside the value.

**What is left** (§8):
- A pose rule that shows the row: a target-derived problem whose rows exercise a + b·c with the accumulator as an
  operand.
- LAWNEAR with depth 2 and the data's constants, at a deeper look.
- For the cube, a rest of the way that keeps x.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md (frozen 09:54 PDT, sha256 `5549c2fdacdbf0643b9e1b3da0dac692c46b21a4464fdc61d1f1889ecde3692d`), before any
run of this study, unchanged:

<!-- PREREG -->
# Composition round 3: pre-registration

Written 2026-10-26 09:54 PDT, before any run of this study. COMPOSE3.md §1 repeats this text unchanged.

## The questions

Round 2 (bar3/scale/compose2/COMPOSE2.md; ad77ad4) got H1's round with book_min less its reduction. Bar 3's sources
supplied the reduction:
- x², posed by the law view (LAWNEAR) and singled out by the subgoal score (SUBGOAL);
- the rest of the way from it, learned by the subgoal's deeper probe (REST_DEEP), accepted with SPELL.

Round 3 asks two things:
1. **The row.** With book_min less its reduction and less its multiply-add row, can bar 3's pieces pose a + b·c, or
   anything that gets the multiply and the round?
2. **Generality.** Does round 2's winning setup, unchanged, get the round for targets of the family that no rule was
   written for: x³ mod N (SP_BREADTH.md's cube) and 10·x² + 1 mod N (its tenrho), at the tiny core's widths?

## Theory, in brief

**The row.**
- What it is: `LDH.Y CLK P[MUL.X INC]`, a + x·d over the places, the multiply's loop body. The multiply is
  `CLA W[row INC SHL.X]`, 9 tokens over the primitives without the row. Every probe here reaches cost 5 at most.
- **The book's own mul law is still posed** without the row: its targets come by repeated addition. But its program is
  those 9 tokens.
- **Bar 3's sources pose from the target's rows only.**
  - The restriction x² < N's target is x² itself, but at the acceptance width (N' < 1,000) x' reaches 31, so only the
    two-digit multiply passes.
  - Its practice probes have one-digit x', and there the cheapest program is a specialised row, 6 tokens
    (`CLA LDH.X CLK P[MUL.X INC]`): out of reach, rejected at width 3, and with its multiplier digit taken from X it
    does not compose into the multiply.
  - No value holding x² exists before the multiply, so LAWNEAR and SUBGOAL have nothing to find.
- **What of CARRY.md §12 carries over, and what does not** (it invented multiply-accumulate at one digit place on bar
  3's counting machine):
  - *widening pins:* the oracle arm's rows pin no register (every probe's registers vary), so nothing is widened;
  - *the loop-view score* needs a widened candidate's step and a register count. This machine's loops run a fixed
    number of times (P places, B the base, W the width), and no posed candidate's step is the row;
  - *conjunctions of restrictions:* the restriction terms (1, 2, 3, X, N and their sums and products) cannot single
    out a one-digit x. On this machine, though, the one-place multiply-accumulate is a primitive, `MUL`, so it needs
    no inventing;
  - *the deeper probe* carries over, and is already in round 2's set (SUBGOAL's, REST_DEEP).

  So nothing new is ported for part 1.

**Generality.** Round 2's route needs a value that is:
- (a) one law of depth 1, over g, X and N, from the target;
- (b) cheap enough for the look's held levels (cost ≤ 3 at the look's 300k states);
- (c) inside the practice registers (P = 2 W places).

x² is all three: x² mod N = mod(x², N), `LDA.X STA.Y T.mul.x,o.XY` costs 3, and x'² fits. Neither new target has one:
- **x³ mod N:**
  - its depth-1 values (x³, or x³ less multiples of N) cost about 8 and overflow the registers, since x'³ needs 3 W
    places;
  - x² is two laws away, mod(mul(g, X), N), and the rest of the way from x² needs x, which a rest's inputs (X = the
    value, N) do not hold: x³ mod N is no function of (x² mod N, N) (SUBGOAL.md §1);
  - and the round needs x to survive a multiply and a reduction (SP_BREADTH.md §6).
- **10·x² + 1 mod N:**
  - its depth-1 values cost at least 5, and 10·x'² + 1 overflows at width 2;
  - x² is three laws away;
  - the law view has no constants here, which bar 3's score takes from the data.

So LAWNEAR finds nothing for either, no value is singled out by a law, and REST_DEEP has nothing to aim at. SPELL is a
property of the search and general, but it acts only once such a value is posed.

## Design (`make_copies3.py`; `hooks3.py`)

Round 2's copies, re-made from the committed files with round 2's replacement lists. Every new change is behind its
own flag; with all of them off the copies are round 2's.
- **TARGET_FN** (`x2`, the default; `x3`; `10x2p1`):
  - the oracle's answers on its draws, and the 400 answer rows' answers at full size, are that function of (x, N),
    from Python's integers;
  - at `x2` the data's own answers are used, as before;
  - a found program is checked as the round is: iterated T times on H1's two ladders' (x, N), against the function
    iterated T times, T = 1 to 64 (`verify3.py ladder`).
- **TIEFOLLOW:** the tie-break finds the rest of the way from the practised subgoal by its value, not by its name. It
  takes the rest root whose starting value, run on the target's probes, equals the subgoal's value there. That is
  round 2's §8 item, cheap here.

## Runs

Each on one B200, round 2's settings (TARGET small, ROUND_TOO, NP 8, deep probes over whole concepts, SECONDS 2,400).
- **Replays, every new flag off:**
  - round 2's B2rd (`c2_b2_rd`) must give its record again, round and all;
  - round 1's book_min and no-practice must give theirs.
- **Part 1, R1** (`c3_r1`): LAWS lt;red;add;mul, PATTERNS pair;x,o (book_min less x + o N and less o,x,d), round 2's
  full set with REST_DEEP and TIEFOLLOW, lpca, LEVELS 10.
- **Part 2, G3 and G10** (`c3_g_x3`, `c3_g_10x2p1`): B2rd's flags and book unchanged, TARGET_FN x3 or 10x2p1, lpca,
  LEVELS 7 (B2rd's round came at level 5).
- **If an arm gets a round:** each of its flags off in turn, where cheap, and the round checked on both ladders to
  T = 64.

Tiny tests first, locally under a memory cap:
- the copies with every new flag off against round 2's tiny records;
- TARGET_FN's draws against Python;
- TIEFOLLOW's match on a library holding x²'s routine.

Money: soft $30, cap $40, `spend.py 4750` before each launch, one spend watcher.

## Predictions

| what | prediction | confidence |
|---|---|---|
| the replays | their records again, apart from times | high |
| R1 | the comparison, the link and the addition practised, at levels 0–2 | high |
| | the multiply never practised, and no row | high |
| | no small target, no round | high |
| G3 (x³ mod N) | the five pieces at levels 0–4, as in B2rd | medium-high |
| | LAWNEAR finds no state | high |
| | no subgoal with a law; no small target, no round | high |
| G10 (10·x² + 1 mod N) | the five pieces at levels 0–4 | medium-high |
| | LAWNEAR finds no state | medium-high |
| | no small target, no round | high |
| TIEFOLLOW's tiny test | it matches the rest of the way from x² posed under a new name | high |
| the conclusion | the route's pieces are x²-shaped in LAWNEAR's reach and law depth; SPELL and REST_DEEP's mechanism are general | medium-high |
<!-- /PREREG -->

### 2. Design

#### 2.1 The copies (`make_copies3.py`)

The committed files with round 2's replacement lists (imported from bar3/scale/compose2/make_copies.py, their module
paths moved to this folder), then round 3's, each asserted to match exactly once:

| copy | from | replacements (round 2's + round 3's) |
|---|---|---|
| `spcore_c3.py` | integration/self_practice/spcore.py | 12 + 1: TIEFOLLOW's default |
| `oracle_c3.py` | integration/self_practice/oracle.py | 4 + 1: the oracle's answers by TARGET_FN |
| `core_c3.py` | integration/self_practice/core.py | 8 + 3: the answer rows by TARGET_FN, TIEFOLLOW, the config |
| `climb_c3.py`, `clean_c3.py` | round 1's climb.py and clean.py | 1 each |
| `hooks.py` | round 2's hooks.py, paths moved | 4: TARGET_FN, TARGET_F, TIEFOLLOW's `follow` and its use |

Round 2's subgoal score (bar3/scale/compose2/subgoal_c2.py) is used as it is. The pre-registration names the hooks
file `hooks3.py`; `make_copies3.py` writes it as `hooks.py`, so that round 2's replacements, which import `hooks`,
find it under the same name.

#### 2.2 The two flags

- **TARGET_FN** (`x2`, the default; `x3`; `10x2p1`):
  - the oracle's answers on its draws, and the 400 answer rows' answers at full size, are that function of (x, N), by
    Python's integers;
  - at `x2` the data's own answers are used, as before;
  - nothing else changes, in the sources or the scores.
- **TIEFOLLOW**: when the practised subgoal's rest is not posed under its name, the tie-break takes the rest root posed
  this level whose starting value, run on the target's probes, equals the subgoal's value there.

### 3. The tests before the arms

| test | where | what it showed |
|---|---|---|
| the copies with every round-3 flag off (`logs/off_*`, `logs/on_*`; `logs/tiny_round3.txt`) | local, tiny budgets | round 2's tiny records, apart from times, with round 2's flags off (book_min, clean values) and on (B1, B2): 4 of 4 |
| smoke runs (`logs/smoke_x3*`, `logs/smoke_10x2p1_r1*`) | local, tiny budgets | the cube on B2rd's book and 10·x² + 1 on R1's, two levels each, every flag on: no errors; the config records `target_fn` |
| `tiny3_test.py` (`logs/tiny3_test.txt`) | local | TARGET_FN x3 and 10x2p1: the draws' answers are the function on all 16 probes; at x2, TIEFOLLOW finds the rest of the way from x² posed under a new name (`rest.3@XN`, from register S of `T.target.rst5.XN`, where the link keeps A before subtracting: x² on every probe), and the tie-break by name finds nothing |
| `verify3.py` at x2 (`logs/verify3_selfcheck_x2.txt`) | local | Python's iterated answers equal the ladders' own (asserted); round 2's round, 768/768 on all 14 rungs |

### 4. The runs (`modal_compose3.py`; `records/*.json` and `.log`)

Each on one B200, launched together at 10:35 PDT after `spend.py 4750` (LAUNCH OK at $3,637.26 billed this month), one
spend watcher (stop at $36; `logs/watch_c3_1.txt`). Round 2's settings: TARGET small, ROUND_TOO, lpca, NP 8, deep
probes over whole concepts, SECONDS 2,400. "B2rd's flags" are CLEAN_VALUES, SUBGOAL, LAWNEAR, ROWCHECK, TIEBREAK,
RESTRICT_ROWS, SPELL=16 and REST_DEEP.

| tag | SOURCES | LAWS / PATTERNS | flags | LEVELS | seconds | GPU peak |
|---|---|---|---|---|---|---|
| `replay_cp_none` | none | | none new | 1 | 4.5 | 12.1 GiB |
| `replay_cp_book_min` | law | book_min's | none new | 10 | 582 | 25.1 GiB |
| `replay_c2_b2_rd` | law and bar 3's | lt;red;add;mul;acc.mul / pair;x,o;o,x,d | B2rd's | 10 | 698 | 24.4 GiB |
| `c3_r1` | law and bar 3's | lt;red;add;mul / pair;x,o | B2rd's and TIEFOLLOW | 10 | 1,325 | 25.7 GiB |
| `c3_g_x3` | law and bar 3's | as B2rd | B2rd's, TARGET_FN x3 | 7 | 701 | 27.5 GiB |
| `c3_g_10x2p1` | law and bar 3's | as B2rd | B2rd's, TARGET_FN 10x2p1 | 7 | 755 | 24.3 GiB |

Every run ended on its own, with exit 0. The replays reproduce their records exactly, apart from times
(`same.py`): round 1's cp_none and cp_book_min, and round 2's c2_b2_rd, whose round came again at level 5. No arm got
a round, so no flag-off arms were run.

### 5. Outcomes (`records/c3_*.json`)

**R1, book_min less its reduction and its row: the multiply never comes.**
- **Levels 0–2:** the comparison, the link and the addition (10 x at level 0).
- **Level 3:** the multiply (mul on x,o) posed in all twelve bindings, its targets by repeated addition. It was
  unsolved by every probe. Its deep probes, focus empty (the primitives alone, 5.8M states), held up to cost 4 and
  streamed cost 5 (17.2–17.9M states). Nothing was learnable.
- **Levels 4, 6, 8:** other bindings of the comparison, the link and the addition.
- **Levels 5 and 7:** nothing.
- **Level 9:** two values. One (`B9.1 T.red.pair.XA`) was singled out by table alone (key (0, 6): 6 rows saved, no
  law), and REST_DEEP probed its rest at 10M states (cost 5 streamed) and found nothing.
- **The end:** the 10-level cap, 1,325 s. LAWNEAR found no state in any level. No small target, no round.

**G3, round 2's setup on x³ mod N: the pieces, and no route.**
- **Levels 0–4:** the comparison, the link, the addition, the row and the multiply, at B2rd's levels.
- **Level 5:** 11 candidates, nothing new learnable.
- **Level 6:** other bindings of the row.
- **The end:** the 7-level cap, 701 s.
- LAWNEAR found no state, the subgoal score singled out nothing, REST_DEEP never fired. No small target, no round.

**G10, round 2's setup on 10·x² + 1 mod N: the pieces, two table subgoals, no route.**
- **Levels 0–4:** the five pieces, beside values built from 10 x and the addition (`T.val.76.tgt.XN` = 10 x plus x).
- **Levels 4 and 5:** the subgoal score singled out a value by table alone (key (0, 2), "g,N": 2 rows saved, no
  law). The law view was on at depths 1 and 2 here, since 10·x² + 1 mod N is no law of X and N of depth 2 or less.
  - One of them, `T.val.101.tgt.XN` = `T.val.76.tgt.XN SHR.A T.red.pair.AN`, is a shortcut that fits a few probes,
    not a piece.
  - REST_DEEP probed both values' rests at 10M states (cost 5 streamed), and found nothing.
- **The end:** the 7-level cap, 755 s. LAWNEAR found no state. No small target, no round.

### 6. Predictions against outcomes

| what | prediction | confidence | outcome |
|---|---|---|---|
| the replays | their records again, apart from times | high | **right**: all three, B2rd's round included |
| R1 | the comparison, the link and the addition at levels 0–2 | high | **right** |
| | the multiply never practised, and no row | high | **right** |
| | no small target, no round | high | **right** |
| G3 | the five pieces at levels 0–4 | medium-high | **right** |
| | LAWNEAR finds no state | high | **right** |
| | no subgoal with a law; no small target, no round | high | **right** (no subgoal at all) |
| G10 | the five pieces at levels 0–4 | medium-high | **right** |
| | LAWNEAR finds no state | medium-high | **right** |
| | no small target, no round | high | **right**; two subgoals by table alone, not foreseen |
| TIEFOLLOW's tiny test | it matches the rest from x² posed under a new name | high | **right** |
| the conclusion | the route is x²-shaped in LAWNEAR's reach and law depth; SPELL and REST_DEEP's mechanism are general | medium-high | **consistent**: nothing here contradicts it, and nothing tests SPELL or REST_DEEP on a target where LAWNEAR finds a value |

### 7. Dead ends, and how hard each was tried

| what | how hard | where it stopped |
|---|---|---|
| the row from bar 3's sources (R1) | 10 levels, 1,325 s on a B200; every round-2 flag, REST_DEEP and TIEFOLLOW; the multiply's deep probes at 5.8M states in each binding | the multiply's 9 tokens are past every probe (cost 5 at most); no source poses the row |
| CARRY.md §12's pieces | read, not ported | they need pinned registers, a register-counted loop and digit-isolating conjunctions, none of which this machine's oracle arm has |
| round 2's setup on x³ mod N (G3) | 7 levels, 701 s | no value one law from the target within the look's reach |
| round 2's setup on 10·x² + 1 mod N (G10) | 7 levels, 755 s; two REST_DEEP probes at 10M | the same; its two subgoals were table shortcuts, their rests unsolved |

### 8. What is left

- **The row needs a pose rule that shows it:** a problem from the target's rows whose solution is a + b·c with the
  accumulator an operand. Nothing here derives one. The book poses it from its own pattern (o, x, d).
- **A deeper LAWNEAR,** with the law view at depth 2 and the data's constants (as bar 3's score has them), and a look
  that holds cost 4. That would see x² for the cube (mod(mul(g, X), N)) and 10 x² for tenrho.
- **For the cube, a rest of the way that keeps x,** since x³ mod N is no function of (x² mod N, N).
- **TIEFOLLOW is tested, but never acted in an arm.**
- **Under lp, a clean link** (round 2's §8).

### 9. Cost (`cost.py`; `logs/cost_c3.txt`, `logs/watch_c3_1.txt`)

The app `ap-NASHTBwgH4BoHelz0KBR52` ran in the hour from 17:00 UTC, not yet billed when this was written. Its six runs
took about 74 B200-minutes with startup:
- the replays: 4.5 s, 582 s and 698 s;
- the arms: 1,325 s, 701 s and 755 s.

That is **about $7.8** at the $6.37 a B200-hour of round 2's billed hours: under the soft $30. The watcher's last
estimate (18:05 UTC, as it exited with the app stopped) was $8.92, an upper bound since it charges the whole hour.
Local work ran under an 8 GB memory cap.

### 10. Files and how to rerun

- `PREREG.md`: read-only, hash in §1.
- `make_copies3.py` writes the copies: `spcore_c3.py`, `oracle_c3.py`, `core_c3.py`, `climb_c3.py`, `clean_c3.py`, and
  `hooks.py` from round 2's. Round 2's subgoal score is imported from its folder.
- `modal_compose3.py`: the arms, one B200 each, app and Volume `bar3-compose3`.
- `tiny3_test.py` (TARGET_FN, TIEFOLLOW), `verify3.py` (a program on H1's ladders for any TARGET_FN), `cost.py`.
  Round 1's `same.py` compares records.
- `records/`: the six runs' logs and records, fetched from the Volume. `logs/`: the local tests, the launch, the
  watcher.

```
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose3.make_copies3
cd cpu_first && H1_DATA=~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2 TARGET_FN=x2 ~/github/sotaku/venv/bin/python -m bar3.scale.compose3.tiny3_test
cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose3/modal_compose3.py --arms replay_cp_none,replay_cp_book_min,replay_c2_b2_rd,c3_r1,c3_g_x3,c3_g_10x2p1
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/compose3/modal_compose3.py --mode fetch
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.same bar3/scale/compose2/records/c2_b2_rd.json bar3/scale/compose3/records/replay_c2_b2_rd.json
cd cpu_first && TARGET_FN=x2 ~/github/sotaku/venv/bin/python -m bar3.scale.compose3.verify3 bar3/scale/compose2/records/c2_b2_rd.json "T.val.126.tgt.XN STA.X T.target.rest.126.XN"
```
