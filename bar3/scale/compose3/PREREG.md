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
