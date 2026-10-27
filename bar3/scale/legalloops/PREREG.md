# Legal sources for the row and the multiply: pre-registration

Written 2026-10-26 21:48 PDT, before any run on a GPU, the tiny test included. LEGALLOOPS.md §1 repeats this
text unchanged and records the time and hash it was frozen with.

## The question

Test 3 (bar3/scale/legalclimb/LEGALCLIMB.md) ran a fully legal climb to H1's round. Its book kept five laws: the
comparison, the link, the addition, the multiply-add row and the multiply. The row and the multiply had no legal
source but the book. Offline, bar 3 supplies both: round 4's one-place cuts pose the row, and round 5's loop source
(bar3/scale/loops/) poses the multiply as the row's loop. But both read the target's law with numpy, and answer with
Python's integers. Here both parts are legal:
- **the reading**, of H1's own answer rows at full size, uses only the learner's routines and the book's constructions;
- **every answer** is a fact, an identity or the learner's routines.

With the book less its reduction and its row, then less its multiply too, does the round come?

## Theory, in brief

- **Answered legally, every cut and loop is a law of the book** (bar3/LEGAL_SOURCES.md §3.13):
  - the row, a + b·c on (o, x, d): d additions of x, then o added;
  - x·d on (x, d): d additions;
  - Horner's step: red on x + d N, by the identity red(x + d N) = x. It is the same function as round 4's red on
    B x + d below B N, which no identity answers before a reduction exists;
  - the multiply, the row's loop, on (x, o): o additions;
  - the reduction, the step's loop, on x + o N: the identity.

  So legally the source is a decision, which laws to pose, and the decision must come from the target.
- **The reading can be legal.** The slice source's reader evaluates candidate laws with numpy. Here every evaluation
  is a routine's:
  - a product is repeated addition, the book's own mul.ladder, where its count is small;
  - a reduction is the learner's link repeated until its comparison says "below".

  H1's answer rows hold 57 of 780 with x below 1,000, enough to test mod(mul(X, X), N) row by row. So the reading needs
  the comparison, the link and the addition as routines, and nothing more.
- **The order.** The comparison (level 0), the link (1), the addition (2). Then the reading, with the row and the step
  learnable at once (level 3). Then the multiply over the row and the reduction over the step (level 4). Then the
  round, 5 tokens or 2 calls over those two.
- **The floor.** Everything above needs the addition, and the book's addition is checked by the link and the
  comparison. So the legal floor is those three laws on the pair. Without the addition, nothing is read and nothing
  is posed.

## Design (`lread.py`, `make_copies6.py`)

- **The copies:** test 3's committed copies (sha256 asserted), paths moved here. With READ set, the pose phase's laws
  are the book's plus the cuts and loops of the law read. The climb keeps the answer rows for the reader. With READ
  unset, the copies run as test 3's.
- **The reader** (`lread.Reader`):
  - leaves X and N; operations add, mul and mod; left-deep chains to depth 2;
  - a law is read when it is computable on at least 4 of the 780 answer rows and gives each one's answer digit for
    digit; the first such law is the reading;
  - it is tried at each level once the addition, the link and the comparison are routines, at most 3 times.
- **What it poses:** for mul, the row (acc.mul on o,x,d), x·d (mul on x,d) and their loop, the multiply (mul on x,o);
  for mod, the step (red on x+dn) and its loop, the reduction (red on x+on). The book's own constructions answer
  them, and a law the book poses itself is left to it.
- **Settings, every arm:** test 3's: lpca, K 4, NP 8, deep probes over whole concepts, probe 1M, deep 5.8M, SEARCH_S
  900, LEVELS 10, SECONDS 2,700, SPELL 16, 780 answer rows, ROWS 4,096, ROUND 4M. With TARGET=rows: test 3's
  sources (look, values, rests, SUBGOAL, REST_DEEP, CLEAN_VALUES, ROWCHECK), the full-size vocabulary's bodies up to 2.
- **The round's check:** both ladders, T = 1 to 64, seen and unseen moduli (compose3/verify3.py).

## Runs

**The tiny test first:** `t_k1`, k1's first 5 levels, one B200.

**Then the arms**, at most 6 B200s at once, in this order:

| arm | book | sources |
|---|---|---|
| `k1` | lt;red;add on pair (less the reduction, the row and the multiply) | the law read legally, its cuts and loops |
| `k2` | the same | and test 3's sources (TARGET=rows) |
| `k3` | lt;red;add;mul on pair;x,o (less the reduction and the row) | the law read legally, its cuts and loops |
| `k0` | lt;red;add on pair | test 3's sources only: no reading (control) |
| `c1` | book_min | none (control) |
| `k1_s1` | as k1 | as k1, seed 1 |
| `k1_less_add` | lt;red on pair (less the addition too) | as k1 |

A round anywhere is checked on both ladders.

## Seen before this was written (local, CPU)

- **The reader** on test 3's g1 climb at the start of its level 3 (the comparison, the link and the addition as
  routines): it read mod(mul(X,X),N) at depth 2, on 57 rows, with 9,977 routine calls, in 3 s.
  - For the book lt;red;add it poses: the row, x·d, the step, the multiply's loop and the reduction's loop.
  - For lt;red;add;mul: the same, less the multiply, which the book poses.
- **The pose phase** with READ on that state: each new law's answers, by the book's constructions, agree with Python's
  integers on its draws, in every binding: 24 bindings of the row, 12 of each of the others.
- **Flags off:** with READ unset and TARGET=rows, two-level tiny runs give test 3's records exactly, apart from times.

## Predictions

| what | prediction | confidence |
|---|---|---|
| t_k1 | reading at level 3 (57 rows); the row and the step at 3; the multiply and the reduction at 4 | medium-high |
| k1 | as t_k1, and the round at level 4 or 5 | medium-high |
| k1: the round's ladders | 768/768 on all 14 rungs | high, given a round |
| k2 | as k1 | medium-high |
| k3 | as k1, the multiply being the book's; the round at level 4 or 5 | medium-high |
| k0 | no row, no multiply, no round in 10 levels | high |
| c1 | the round at level 5 | high |
| k1_s1 | as k1 | medium-high |
| k1_less_add | no reading (no addition); nothing past the comparison and the link; no round | high |
| the floor | the comparison, the link and the addition; the rest from the target, legally | medium-high |
| money | about $12–20 in all | medium |

## Money

Soft $100, cap $150. `integration/ablation_s3/spend.py 4750` runs before each launch. One spend watcher for this
study's apps, its total the cap less one hour of the containers' cost: 6 B200s at $6.37 an hour, so it stops
everything at $110. Every app is stopped when its runs end.
