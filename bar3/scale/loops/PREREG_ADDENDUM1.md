# Composition round 5, addendum 1: the book with less again, pre-registered

Written 2026-10-26, after the pre-registration's eight arms and before any run below. LOOPS.md §1 records its time and
hash.

## What the eight arms showed

**x² got the round in every arm that kept the comparison and the loops,** each right on both ladders to T = 64:
- `l5_x2`, the book lt;red;add: the multiply came from its loop at level 1 and the reduction from its loop at level 2;
  the round came at level 2;
- `l5_x2_bare`, round 2's search fixes off;
- `l5_x2_less_add`, the book lt;red;
- `l5_x2_less_link`, the book lt alone.

**The other arms:**
- `l5_x3`: the reduction came as a routine from its loop (level 2), and the multiply came from the book's mul at level
  4. No round by the deadline (level 6).
- `l5_10x2p1`: the reduction came from its loop (level 2). The book's mul was never posed, since its answers need the
  book's addition on the pair as a routine, and none was practised. No multiply, no round.
- `l5_x2_lessons` (anchored pairs): the row was not picked in its 3 levels, and there was no round.
- `replay_s4_x2`: round 4's record again.

## Arms

Each on one B200, the pre-registration's settings, all launched together:
- `l5_x2_no_book`: no book at all (SOURCES bar 3's four only), with the cuts and their loops;
- `l5_x2_all_off`: `l5_x2_bare` with round 2's other three flags off too (CLEAN_VALUES, ROWCHECK, RESTRICT_ROWS),
  so every one of round 2's flags is off;
- `l5_x2_bare_no_redloop`: `l5_x2_bare` with the reduction's loop off (LOOPS=mul);
- `l5_x3_less_mul` and `l5_10x2p1_less_mul`: the book lt;red;add (as `l5_x2`), x³ and 10·x² + 1, so that the
  multiply too comes from its loop.

Any round is checked on both ladders to T = 64.

## Predictions

| what | prediction | confidence |
|---|---|---|
| l5_x2_no_book | no round: nothing poses the comparison, so neither the step (7 tokens over the primitives) nor the reduction is learnable | high |
| l5_x2_all_off | the round | medium-high |
| l5_x2_bare_no_redloop | no round: without the reduction's loop and without round 2's search fixes, nothing finds the reduction | medium |
| l5_x3_less_mul | the multiply from its loop by level 2, the reduction by level 3; no round | medium (no round) |
| l5_10x2p1_less_mul | the multiply and the reduction; no round | high (no round) |

## Money

Five B200s at once. The watcher's total is the cap ($80) less one hour of five: stop all at $48.
