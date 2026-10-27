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
