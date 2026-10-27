# Composition round 6, addendum 1: the arms the deadline cut, given 7,200 s, pre-registered

Written 2026-10-27, after the pre-registration's eleven arms and before any run below. READING.md §1 records its time
and hash.

## What the eleven arms showed

**x² with no book got the round,** right on both of H1's ladders to T = 64:
- `r6_x2_nobook`: the comparison from the step's cuts at level 0; the step, the link and the multiply at level 1;
  the reduction from its loop at level 2; the round at level 2 (548 s);
- `r6_x2_nobook_cmp`, the comparison alone: the round at level 2 (423 s);
- `r6_x2_nobook_bare`, every one of round 2's flags off: the round at level 2, by calls (587 s);
- `r6_x2_nobook_link`, the link alone: the link was never learned, so there was no step and no round (the deadline in
  level 7).

**x³ with the sub-law got the round:** mod.mul at level 3, then the round, two calls of it (`LDA.X STA.Y
T.mod.mul.x,o,n.XYN STA.X T.mod.mul.x,o,n.XYN`). With the book lt;red;add it came at 2,346 s, and with no book at
2,283 s, each right on both ladders.

**10·x² + 1 with its constant** (`r6_10x2p1`, `r6_10x2p1_nobook`, the same picks through level 4):
- make 1 at level 0, from the primitives;
- x + 1 at level 1, in two bindings (x in X or in N, N in Y);
- the reduction at level 2;
- then values built on x + 1; x² mod N never came as a value;
- the deadline in level 5, no round.

**Anchored pairs** (`r6_x2_lessons_prior`, `r6_x2_lessons_nobook`):
- every look was null again, so every gain tied;
- the row came at level 0, with the prior and with no book;
- at level 1 the tie order took more of the row's bindings;
- at about 1,100 s a level, the deadline came in level 2.

**The replay** gave round 5's record again, apart from times.

## Arms

Each on one B200, launched together. The pre-registration's settings, with SECONDS 7,200 (the runner's container
timeout raised to 9,000 s):
- `r6_10x2p1_nobook_long`: `r6_10x2p1_nobook`. The book arm took the same picks through level 4, so one long arm
  stands for both;
- `r6_x2_lessons_prior_long`: `r6_x2_lessons_prior`;
- `r6_x2_lessons_nobook_long`: `r6_x2_lessons_nobook`.

Any round is checked on both ladders to T = 64.

## Predictions

| what | prediction | confidence |
|---|---|---|
| r6_10x2p1_nobook_long | no round by level 10. x + 1 is practised only in bindings that make the round 8 tokens or more, and lpca poses no practised concept's other bindings | medium |
| r6_x2_lessons_prior_long | the multiply by level 4; the step later, since its comparison's bindings are book laws, which the prior ranks last; no round by the deadline | low-medium |
| r6_x2_lessons_nobook_long | the multiply by level 4; no round by the deadline: the anchors stay on the costliest concepts' bindings, and the comparison comes last in pool order | low-medium |

## Money

- Three B200s. The watcher's total is the cap ($140) less one hour of three: stop all at $120.
- This study so far: about $35. With these, about $75, under the soft $100.
