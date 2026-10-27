# Legal sources for the row and the multiply

*Written 2026-10-26. Eight B200 runs on Modal (one tiny test, seven arms), about $7 (§7). Code in this folder; records
in `records/`, logs in `logs/`. Nothing was committed.*

<!-- PLAIN -->
## In plain words

**The question.** Test 3 (bar3/scale/legalclimb/LEGALCLIMB.md) got H1's round from a fully legal climb, but its book
kept five laws: the comparison, the link, the addition, the multiply-add row and the multiply. The row and the multiply
had no legal source but the book. Offline, bar 3 has sources for both: round 4's one-place cuts pose the row, and round
5's loop source (bar3/scale/loops/) poses the multiply as the row's loop. But both read the target's law with numpy,
and answer with Python's integers. Here both parts are legal:
- **the reading** of H1's own answer rows, at full size, uses only the learner's routines and the book's constructions;
- **every answer** is a fact, an identity or the learner's routines.

With the book less its reduction and its row, then less its multiply too, does the round come?

**What came: yes. The book can shrink to three laws on the pair: the comparison, the link and the addition.** Every
arm that reads the target legally got H1's round at level 4, a level before book_min, and every round is right on
both ladders.

| arm | the book | sources | the round |
|---|---|---|---|
| `k1` | lt;red;add on the pair | the law read legally: its cuts and loops | level 4, 442 s |
| `k1_s1` | the same | the same, seed 1 | level 4, 536 s |
| `k2` | the same | the same, and test 3's sources on the answer rows | level 4, 438 s |
| `k3` | book_min less its reduction and its row | the law read legally: its cuts and loops | level 4, 577 s |
| `k0` | lt;red;add on the pair | test 3's sources only, no reading (control) | none in 10 levels |
| `c1` | book_min (control) | none | level 5, 496 s |
| `k1_less_add` | lt;red on the pair | the law read legally | none in 10 levels: nothing read |
| `t_k1` (the tiny test) | as k1, 5 levels | as k1 | level 4, 546 s |

All eleven rounds (two an arm where the multiply came in both bindings) are right on both of H1's ladders: 768 of 768
on all 14 rungs, T = 1 to 64, seen and unseen moduli.

**How the round comes** (k1; t_k1, k1_s1, k2 and k3 the same):
1. **Levels 0–2, the book's three laws:** the comparison, the link and the addition, answered by facts and the
   learner's routines, as in test 3.
2. **Level 3, the reading.** With those three as routines, the reader tries laws on H1's 780 answer rows at full
   size: chains of add, mul and mod over X and N, depth 1, then 2. Every value in it is a routine's output:
   - a product is repeated addition, the book's own construction of the multiply;
   - a reduction is the link, repeated until the comparison says "below".

   It reads `mod(mul(X,X),N)`, exact on all 57 rows where it can be computed (x below 1,000), in 9,977 routine calls
   and 11–24 s.
3. **Level 3, what the reading poses:** the multiply's cuts (the row a + b·c, and x·d) and its loop (the multiply);
   the reduction's cut (Horner's step, red on x + d·N) and its loop (the reduction). Each is a law of the book,
   answered by the book's own constructions. The row (`LDH.Y CLK P[MUL.X INC]`) and the step (`B[T.red.pair.AN]`,
   the link ten times) are learned at once.
4. **Level 4:** the multiply, a loop over the row (`CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]`); the reduction, Horner's
   loop over the step (`LDA.N P[BRG.X T.red.x+dn.AN]`); and x·d.
5. **The round**, 5 tokens over those two, by the round's own search at level 4:
   `LDA.X STA.Y T.mul.x,o.YX STA.X T.red.x+on.XN`.

**The floor: the comparison, the link and the addition, on the pair.**
- **Without the reading** (k0), the same three laws get nothing more: no row, no multiply, no round in 10 levels.
  Test 3's look tried 9,154 to 22,773 values a level, and none passed: with no multiply, no value is x².
- **Without the addition** (k1_less_add), nothing is read, so nothing is posed. The climb practised only other
  bindings of the comparison and the link, to the level cap.
- **With the multiply left in the book** (k3), the reading poses the rest, and the round comes at level 4 too.
- **With book_min** (c1), the round comes at level 5. The cuts make the climb a level shorter: the step is learned
  with the row at level 3, so the reduction is a 4-token loop over it at level 4. book_min finds the reduction whole,
  5 tokens over the link, by a deep probe at level 5.

**What it does not show.**
- **The reader's grammar is designed.** Its leaves (X, N), its operations (add, mul, mod), its depth (2) and the order
  of its tries are chosen by hand. So is the table of what each operation cuts to and loops over. What is legal now
  is every number it computes and every decision it makes: the target decides, through the learner's routines, which
  of the book's laws to pose. Their forms, and the constructions that answer them, are still the book's.
- **57 rows.** The reading rests on the answer rows with x below 1,000, where a product's count is small. The round
  itself is checked on every row of both ladders.
- **One formula.** mod(mul(X,X),N) is what this grammar says at depth 2. Other formulas need other chains.
- **The tiny core, not a training run**, as in test 3.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md was frozen (read-only) at 21:48:04 PDT, before any run on a GPU; the tiny test was spawned at 21:48:09. Its
sha256 is `61828110…f3ea`; `logs/prereg_sha256.txt` holds it with the code's hashes. Its header's time, 21:48, is the
freeze time. No arm was added, so there is no addendum. Its text, unchanged:

<!-- PREREG -->
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
<!-- /PREREG -->

### 2. The legal reader (`lread.py`) and the copies (`make_copies6.py`)

- **The copies:** test 3's committed copies at ef10ac1 (sha256 asserted), module paths moved here, and four asserted
  replacements, each marked "legalloops":
  - `hooks.py`: READ from the environment; with READ set, the pose phase's laws are the book's plus the read law's
    cuts and loops (`lread.pose`);
  - `core_k.py`: the climb keeps the 780 answer rows for the reader; READ in the run's config.

  With READ unset the copies run as test 3's.
- **The reader** (`lread.Reader`): values are digit strings at full size (16 places), each with the mask of the rows
  on which it was computed.
  - add: the learner's addition on the pair;
  - mul(a, b): b additions of a, where b, read off its digits as a count, is below 1,000. The book's own multiply
    construction (mul.ladder, `spcore_k.repeat_add`) reads its count the same way;
  - mod(a, b): the learner's link, repeated until the learner's comparison says the value is below b, at most 1,000
    times.

  A law is read when it is computable on at least 4 rows and gives each one's answer digit for digit. The first that
  does, by depth, then inner chain, then operation, is the reading. The reader runs once the addition, the link and
  the comparison are routines on the pair, at most 3 times.
- **What the reading poses** (`lread.CUTS`, `lread.LOOP_OF`), for each operation in the law read:
  - mul: the row (acc.mul on o,x,d), x·d (mul on x,d), and their loop, the multiply (mul on x,o);
  - mod: Horner's step (red on x+dn), and its loop, the reduction (red on x+on);
  - add: nothing.

  A law the book poses itself is left to the book. The book's own constructions answer the rest (spcore's
  law_target: identities and the learner's routines), as they answer any law of the book.
- **The record:** each reading appends a `task: read` record (level, law, rows, routine calls, the laws posed).

### 3. Checks before the runs (local, CPU)

- **The reader** on test 3's g1 climb at the start of its level 3 (`tiny.py`, MODE=read): `mod(mul(X,X),N)` at
  depth 2, on 57 rows, in 9,977 routine calls and 3 s. For lt;red;add it poses the row, x·d, the step, the multiply
  and the reduction; for lt;red;add;mul the same less the multiply.
- **The pose** on that state (`tiny.py`, MODE=pose): every new law's answers, by the book's constructions, agree with
  Python's integers on its draws, in every binding (24 bindings of the row, 12 of each of the others).
- **Flags off:** READ unset, TARGET=rows, book_min, two levels at the core's local sizes: this study's copies and test
  3's give the same records apart from times (`logs/same/`, compared by bar3/scale/legalclimb/same.py).

### 4. The tiny test (one B200)

t_k1, k1's first 5 levels:

| level | practised | ends at |
|---|---|---|
| 0 | the comparison, lt on the pair (AX, AY, AN) | 36 s |
| 1 | the link, red on the pair | 59 s |
| 2 | the addition, add on the pair | 78 s |
| 3 | the reading (21 s); the row (4 bindings, cost 5); the step (3 bindings, cost 2) | 251 s |
| 4 | the multiply (XY, YX); the reduction (XY, XN); x·d (4 bindings); the round (cost 5) | 546 s |

Its two rounds, `LDA.X STA.Y T.mul.x,o.YX STA.X T.red.x+on.XN` and the same over `T.mul.x,o.XY`, are 768/768 on all
14 rungs.

### 5. The arms

**The levels, in every arm that reads** (k1, k1_s1, k2, k3, as t_k1):

| level | practised | found by |
|---|---|---|
| 0 | the comparison, lt on the pair | the regular probe, cost 4 |
| 1 | the link, red on the pair | cost 3 |
| 2 | the addition, add on the pair | cost 4 |
| 3 | the reading; the row, acc.mul on (o, x, d), 4 bindings; the step, red on x + d·N, 3 bindings | cost 5; cost 2 |
| 4 | the multiply on (x, o); the reduction on x + o·N; x·d | the deep probe over the row, cost 5; cost 4; 2–4 |

k1_s1 learned the multiply in one binding (XY), as `CLA W[T.acc.mul.o,x,d.AYX INC SHL.Y]`, so it has one round. The
routines are spelled the same in k1, k2, k3 and t_k1.

**Times** (seconds from the start):

| | k1 | k1_s1 | k2 | k3 | t_k1 | c1 |
|---|---|---|---|---|---|---|
| level 2 ends | 61 | 58 | 62 | 83 | 78 | 88 |
| the reading (in level 3's pose) | 11.5 | 24.2 | 12.4 | 22.5 | 21.3 | none |
| level 3 ends | 190 | 239 | 188 | 267 | 251 | 209 |
| level 4 ends | 442, the round | 536, the round | 438, the round | 577, the round | 546, the round | 325 |
| level 5 ends | | | | | | 496, the round |

**What differed:**
- **k2** (test 3's sources too): its look ran at levels 2–4, with the link and then the step as its chain (9,154 to
  26,686 values); nothing passed, because no value is x² before the multiply is a routine. The round came at level 4
  from the round's own search, as in k1.
- **k3** (the multiply in the book): the reading posed the row, x·d, the step and the reduction, and left the
  multiply to the book. The same levels as k1.

**The controls.**
- **c1, book_min alone:** the row at level 3, the multiply at level 4 (the deep probe over the row), the reduction at
  level 5 (the deep probe over the link, cost 5: `LDA.N P[BRG.X B[T.red.pair.AN]]`), the round at level 5. Both its
  rounds are 768/768 on all 14 rungs.
- **k0, lt;red;add with test 3's sources and no reading:** the three laws by level 2. Then level 3 posed nothing, and
  levels 4, 6 and 8 only other bindings of the three. The look, with the link as its chain, tried 9,154 to 22,773
  values at levels 2–9; none passed. The level cap at 139 s.
- **k1_less_add, lt;red with the reading:** the comparison and the link by level 1. The reader never ran (no
  addition), so no read record and nothing posed. Levels 3, 5, 7 and 9 practised other bindings of the two. The
  level cap at 107 s.

### 6. Predictions against outcomes

| prediction | outcome | |
|---|---|---|
| t_k1: the reading at level 3 (57 rows); the row and the step at 3; the multiply and the reduction at 4 | so | held |
| k1: as t_k1, and the round at level 4 or 5 | level 4 | held |
| k1: the round's ladders 768/768 on all 14 rungs | so, both rounds | held |
| k2: as k1 | level 4 | held |
| k3: as k1, the multiply being the book's; the round at level 4 or 5 | level 4 | held |
| k0: no row, no multiply, no round in 10 levels | so | held |
| c1: the round at level 5 | so | held |
| k1_s1: as k1 | level 4 (one round: the multiply in one binding) | held |
| k1_less_add: no reading; nothing past the comparison and the link; no round | so | held |
| the floor: the comparison, the link and the addition; the rest from the target, legally | so | held |
| money: about $12–20 in all | about $7 | below the range |

Every pre-registered outcome held. The money came in under its range: every arm ended in 2–10 minutes, at level 4,
where the estimate allowed for level 5 and longer searches.

### 7. Cost

- **Billed** (`cost.py`, at 05:10 UTC): $1.65 through the 04:00 UTC hour: the tiny test $1.15, the arms' first minute
  $0.50. The 05:00 UTC hour, the arms' other nine minutes, was not in the report yet.
- **By the runs' own times:** 3,279 s of module time on B200s (55 minutes) at $6.37 an hour is $5.80. The tiny test
  was billed 1.19 times its module time (start-up included); at that rate the arms come to about $5.80 and the whole
  to about $7.
- **The budget:** soft $100, cap $150. The spend watcher's stop-all was $110; it never acted. `spend.py 4750` said
  LAUNCH OK before both launches; the month's bill was $3,850.85 at 05:14 UTC.
- **Apps:** all stopped: the tiny test's, the arms', and the fetches'.

### 8. What it shows, and what it does not

- **The row and the multiply have a legal source: the target, read by the learner's own routines.** LEGAL_SOURCES.md
  §3.13 found the slice reader's legal version would have to wait for a multiply. This reader needs no multiply
  routine: its products are the book's own construction, repeated addition by the learner's addition, the same that
  answers the multiply's practice. So it reads at level 3, before any multiply exists.
- **The legal floor is three laws on the pair:** the comparison, the link and the addition. From them the target,
  read legally, poses the row, x·d, the step, the multiply and the reduction, and the round comes at level 4 in every
  arm that reads: k1, k1_s1, k2, k3 and the tiny test.
- **Each part is needed.** Without the reading the three laws stop (k0). Without the addition there is no reading
  (k1_less_add).
- **The cuts shorten the climb.** With the step posed, the reduction is a 4-token loop found by the regular probe at
  level 4. book_min's reduction is 5 tokens and needs the deep probe at level 5.
- **Not shown:**
  - the reader's grammar and its table of cuts and loops, which are designed: the decision is legal, the forms are
    the book's;
  - other formulas: depth-2 chains of add, mul and mod over X and N;
  - the climb inside a training run (an sp2-style file: rows kept in buffers, judge question 3);
  - the choosing rule (lpca) and the search's sizes, which stay designed.

### 9. Files and how to rerun

- `lread.py`: the reader and the pose. `make_copies6.py`: the copies (`spcore_k.py`, `climb_k.py`, `core_k.py`,
  `hooks.py`, `oracle_k.py`, `clean_k.py`, `rowclimb.py`, `lclimb.py`).
- `tiny.py`: the local checks (MODE=read, MODE=pose). `modal_lloops.py`: the launcher (Volume `bar3-legalloops`).
  `cost.py`: §7.
- `PREREG.md`: frozen. `logs/prereg_sha256.txt`: its hash and the code's.
- `records/<arm>.json`, `records/<arm>.log`: every run. `logs/ladder_<arm>_<binding>.txt`: the rounds on both
  ladders. `logs/same/`: the flags-off check's two records.
- `logs/*_app.log`: the apps' logs as captured; `logs/watch_*.txt`: the spend watchers.

From `cpu_first`, with `PY=~/github/sotaku/venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1` and `H1_DATA` exported:

```bash
$PY -m bar3.scale.legalloops.make_copies6
READ=1 MODE=read $PY -m bar3.scale.legalloops.tiny
READ=1 MODE=pose $PY -m bar3.scale.legalloops.tiny
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach bar3/scale/legalloops/modal_lloops.py --arms t_k1
$PY -m modal run --detach bar3/scale/legalloops/modal_lloops.py --arms k1,k2,k3,k0,c1,k1_s1,k1_less_add
$PY -m modal run bar3/scale/legalloops/modal_lloops.py --mode fetch
$PY -m bar3.scale.compose3.verify3 bar3/scale/legalloops/records/k1.json "LDA.X STA.Y T.mul.x,o.YX STA.X T.red.x+on.XN"
$PY bar3/scale/legalloops/cost.py
```
