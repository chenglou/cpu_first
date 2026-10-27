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
