# Composition round 7, addendum 2: a round must give the answers of every T, pre-registered

Written 2026-10-27, after addendum 1's eleven arms and before any run below. FAMILY.md §1 records its time and hash.

## What addendum 1's arms showed

**The six tasks that came in phase 1 came again with both fixes** (CONST1 and HOLES): x², the cube, ten, rho, plus3
and tenrho. Each is right on both ladders and on every modulus at x = 0, at a random x and at x = N − 1. CONST1's
replays gave r7_x2's and r7_x3's records again, apart from times.

**minus1 found rounds, but not right ones:**

| arm | the round search | its first program, kept as the round |
|---|---|---|
| `r7_minus1_both` | 7 programs at level 3 (1,194 s) | right on 5 of 14 rungs, 0 of 747 moduli at x = 0 |
| `r7_minus1_const1` | 6 at level 4 (2,299 s) | the same |
| `r7_minus1_holes` | none (the deadline in level 6): nothing made a 1 | – |

- **The first program ends with a reduction of X and a subtraction** (`... T.red.x+on.XN T.lt.pair.AY`). It is wrong
  exactly when a round's input is 0, as SP_BREADTH's constants change was.
- **The training data shows the failure.** H1's training rows have T = 1, 2, 4, 8 and 16, and minus1's rounds pass
  through 0 in 24 of the 33,333 rows of T = 4 and 61 of T = 8. The climb's round search checks only the answers of
  T = 1.
- **Later programs of the same search are right everywhere,** for example `LDA.X STA.Y T.w.pred.add.mul.w2.XYN STA.X
  T.red.x+on.XN`: the window pred(add(mul(a,b),N)), then the whole reduction.
- **ROUNDT's filter, applied offline to the found rounds** (`roundt_offline.py`, `logs/roundt_offline.txt`):
  - it keeps the last two (CONST1) and three (both fixes) of minus1's;
  - it keeps all of the cube's, rho's and tenrho's.

## The fix, behind its own flag

**ROUNDT:** a round must also give the data's answers at every one-digit T past 1. Run T times at full size, it must
match every training row with T = 2, 4 or 8. For x², those are the data's own answers; for another target, its
function applied T times, as its T = 1 answers are. Of the programs found, the first that does is kept. If none
does, the climb goes on. It names no task: it asks a round to be the round on all the data's rows, not only on those
of T = 1. ROUNDT also asks the small target's program to be right at full size on the T = 1 answers.

## Arms

Each on one B200, addendum 1's setup (SECONDS 3,600), launched together, at most 8 at once:
- `r7_minus1_all`: minus1 with all three fixes (CONST1, HOLES, ROUNDT);
- `r7_minus1_const1_roundt`: minus1 with CONST1 and ROUNDT;
- `r7_x2_roundt` and `r7_x3_roundt`: ROUNDT's replays, on phase 1's setup;
- the family with all three fixes: `r7_x2_all`, `r7_x3_all`, `r7_ten_all`, `r7_rho_all`, `r7_plus3_all` and
  `r7_tenrho_all`.

Any round is checked on both ladders to T = 64, and on every modulus at x = 0, at a random x and at x = N − 1.

## Predictions

| what | prediction | confidence |
|---|---|---|
| r7_minus1_all | the round at level 3, the fifth of its search's seven programs, right on both ladders and at x = 0 | high |
| r7_minus1_const1_roundt | the round at level 4, the fifth of six, right on both ladders and at x = 0 | high |
| r7_x2_roundt, r7_x3_roundt | r7_x2's and r7_x3's records again, apart from times and ROUNDT's own records | high |
| r7_*_all, the six others | each round again, at its r7_*_both level, right on both ladders | high |

## Money

- Ten arms, at most 8 at once. A new watcher, over the three apps, stops all at $79 (the cap less one hour of 8
  B200s).
- This study so far: about $45. With these, about $70, under the soft $90.
