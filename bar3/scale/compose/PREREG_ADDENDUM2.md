# The composition test: a seventh arm, pre-registered before its launch

Written 2026-10-26 06:46 PDT. PREREG.md's five arms and PREREG_ADDENDUM.md's `cp_bar3_clean` were launched. All but
`cp_bar3_minus_red` (A2) had ended; A2 was at level 6 of 10. This arm was not launched. COMPOSE.md §1 repeats this
text unchanged.

## Why

A2 is book_min without the reduction's pattern, plus bar 3's sources. It practised the comparison, the link, the
addition, the row and the multiply at levels 0–4, as book_min did, and no reduction at level 5. Its near misses after
the multiply (the looks at the poses of levels 5 and 6) are all built from the near-miss shortcut routines it
practised beside the book's pieces:
- each of the best 12 calls one, and fits 10 of the 16 probes;
- x² (`LDA.X STA.Y T.mul.x,o.XY`) is right only where x'² < N'. No near miss of the run calls the multiply;
- so the rest of the way from x², the one route to the reduction that the theory leaves open (COMPOSE.md §2.1), was
  never posed.

Those shortcut routines read registers the inputs leave dirty (`digit_test.py`: none of A1's 23 is clean).

## The arm

`cp_bar3_minus_red_clean`: A2 with CLEAN_VALUES=1 (`clean.py`). A near miss's value is posed, and the rest of the way
from it, only when it is a function of x' and N'. Everything else is A2's: one B200, RULE=lpca, NP 8, the deep probes
over whole concepts, LEVELS 10, SECONDS 2,400, the small target and the round's search each level.

## Predictions

| prediction | confidence |
|---|---|
| the comparison, the link, the addition, the row and the multiply, at levels 0–4 as in book_min | medium-high |
| after the multiply, a near miss holds x², or x² less a multiple of N, among the best 12 at some level's pose | medium |
| the rest of the way from such a value is posed, and is the reduction at the practice widths | certain, given the line above |
| that rest is learned: a reduction accepted at width 3 | low-medium |
| the small target or the round comes | low |
| if the round comes, it is right on every rung of both ladders | high, given the round |
